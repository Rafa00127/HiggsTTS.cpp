#!/usr/bin/env python3
"""
Convert bosonai/higgs-audio-v3-tts-4b (HuggingFace safetensors) → GGUF F16
for the CrispASR `higgs-tts` backend.

Usage:
    python convert-higgs-tts-to-gguf.py \
        --input bosonai--higgs-audio-v3-tts-4b \
        --output higgs-audio-v3-tts-4b.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    """Map Higgs safetensors name → GGUF name for C++ runtime."""

    n = hf_name

    # ── Backbone (Qwen3-4B) ─────────────────────────────────────────────
    # tied.embedding.text_embedding.*  → token_embd.*
    # body.layers.*                    → blk.*
    # body.norm.*                      → output_norm.*
    # tied.head.text_head.*            → output.*
    n = n.replace("tied.embedding.text_embedding.", "token_embd.")
    n = n.replace("body.layers.", "blk.")
    n = n.replace("body.norm.", "output_norm.")
    n = n.replace("tied.head.text_head.", "output.")

    # Per-layer shortenings (apply to blk.*)
    n = n.replace(".self_attn.q_proj.", ".attn_q.")
    n = n.replace(".self_attn.k_proj.", ".attn_k.")
    n = n.replace(".self_attn.v_proj.", ".attn_v.")
    n = n.replace(".self_attn.o_proj.", ".attn_output.")
    n = n.replace(".self_attn.q_norm.", ".attn_q_norm.")
    n = n.replace(".self_attn.k_norm.", ".attn_k_norm.")
    n = n.replace(".input_layernorm.", ".attn_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".ffn_gate.")
    n = n.replace(".mlp.up_proj.", ".ffn_up.")
    n = n.replace(".mlp.down_proj.", ".ffn_down.")

    # ── Fused modality embedding / head ──────────────────────────────────
    n = n.replace("tied.embedding.modality_embeddings.0.embedding.", "fused_embed.")
    n = n.replace("tied.head.modality_heads.0.", "fused_head.")

    # ── Codec (whole subtree) ────────────────────────────────────────────
    n = n.replace("tied.embedding.modality_embeddings.0.model.", "codec.")
    # Shorten common codec patterns
    n = n.replace("acoustic_encoder.", "ac_enc.")
    n = n.replace("acoustic_decoder.", "ac_dec.")
    n = n.replace("semantic_model.", "sem.")
    n = n.replace("encoder_semantic.", "enc_sem.")
    n = n.replace("decoder_semantic.", "dec_sem.")
    n = n.replace("quantizer.quantizers.", "quant.")
    n = n.replace("feature_extractor.", "fe.")
    n = n.replace("feature_projection.", "fp.")

    # Long sub-path shortenings (keep under 64 char GGUF limit)
    n = n.replace("conv_layers.", "cv.")
    n = n.replace("encoder.layers.", "enc.")
    n = n.replace("feed_forward.intermediate_dense", "ffn1")
    n = n.replace("feed_forward.output_dense", "ffn2")
    n = n.replace("attention.q_proj", "attn_q")
    n = n.replace("attention.k_proj", "attn_k")
    n = n.replace("attention.v_proj", "attn_v")
    n = n.replace("attention.out_proj", "attn_out")
    n = n.replace(".layer_norm.", ".ln.")
    n = n.replace(".final_layer_norm.", ".fin_ln.")
    n = n.replace(".conv_blocks.", ".blk.")
    n = n.replace("res_units.", "ru.")
    n = n.replace("parametrizations.", "pm.")
    n = n.replace("original", "orig")
    n = n.replace("encoder.layer_norm.", "enc.ln.")
    n = n.replace("pos_conv_embed.", "pce.")

    # Detect unmapped prefixes
    if not any(n.startswith(p) for p in (
        "token_embd.", "blk.", "output_norm.", "output.",
        "fused_embed.", "fused_head.", "codec."
    )):
        return f"__UNMAPPED__:{hf_name}"

    return n


# ---------------------------------------------------------------------------
# Weight transformation helpers
# ---------------------------------------------------------------------------

def is_conv1d_weight(name: str, shape: tuple) -> bool:
    """Check if tensor is a regular Conv1d weight (PyTorch [OC,IC,K], written as-is)."""
    if len(shape) != 3:
        return False
    if "conv_t" in name:
        return False  # ConvTranspose1d, handled separately
    if "conv" not in name:
        return False
    if shape[2] <= 1:
        return False  # Not a conv kernel
    return True


def is_convt_weight(name: str, shape: tuple) -> bool:
    """Check if tensor is a ConvTranspose1d weight (needs wperm)."""
    if len(shape) != 3:
        return False
    if "conv_t" in name:
        return True  # Explicitly named conv_t
    return False


def is_snake_alpha(name: str, shape: tuple) -> bool:
    """Detect Snake1d alpha: [1, C, 1] → squeeze to [C]."""
    return "snake" in name and len(shape) == 3 and shape[0] == 1 and shape[2] == 1


def transform_snake_alpha(t: np.ndarray) -> np.ndarray:
    """Snake1d alpha [1, C, 1] → [C] F32."""
    return np.ascontiguousarray(t.squeeze().astype(np.float32))


def is_conv1d_weight(name: str, shape: tuple) -> bool:
    """Check if tensor is a regular Conv1d weight (PyTorch [OC,IC,K], written as-is)."""
    if len(shape) != 3:
        return False
    if "conv_t" in name:
        return False  # ConvTranspose1d, handled separately
    if "conv" not in name:
        return False
    if shape[2] <= 1:
        return False  # Not a conv kernel
    return True


def transform_conv1d_to_ggml(t: np.ndarray) -> np.ndarray:
    """PyTorch Conv1d [OC, IC, K] → written as-is; ggml reads ne=[K,IC,OC].
    Memory layouts match due to torch row-major ↔ ggml col-major duality."""
    return np.ascontiguousarray(t.astype(np.float32))


def transform_convt_wperm(t: np.ndarray) -> np.ndarray:
    """PyTorch ConvTranspose1d [IC, OC, K] → w_perm [K*OC, IC] for GGUF.

    GGML side creates tensor with ne0=IC, ne1=K*OC.
    """
    IC, OC, K = t.shape
    # Reshape to [IC, K*OC] then transpose to [K*OC, IC]
    return np.ascontiguousarray(t.reshape(IC, K * OC).T)


# ---------------------------------------------------------------------------
# Tokenizer extraction from tokenizer.json (HF tokenizers format)
# ---------------------------------------------------------------------------

def extract_tokenizer_info(tokenizer_path: str) -> tuple[list[str], list[str]]:
    """Extract vocab list and merges list from a HF tokenizers tokenizer.json."""
    with open(tokenizer_path, encoding="utf-8") as f:
        tj = json.load(f)

    model = tj.get("model", {})
    vocab_dict = model.get("vocab", {})
    merges_raw = model.get("merges", [])

    # vocab → sorted by id
    max_id = max(vocab_dict.values()) if vocab_dict else -1
    tokens = [""] * (max_id + 1)
    for tok, idx in vocab_dict.items():
        if idx < len(tokens):
            tokens[idx] = tok

    # merges → list of strings
    merges = []
    for m in merges_raw:
        if isinstance(m, (list, tuple)):
            merges.append(f"{m[0]} {m[1]}")
        elif isinstance(m, str):
            merges.append(m)

    return tokens, merges


def extract_special_token_ids(tokenizer_config_path: str) -> dict[str, int]:
    """Extract key special token IDs from tokenizer_config.json for metadata."""
    with open(tokenizer_config_path, encoding="utf-8") as f:
        tc = json.load(f)

    result = {}
    added_tokens = tc.get("added_tokens_decoder", {})
    if not added_tokens:
        # Fallback: use extra_special_tokens and look them up from tokenizer.json
        return result

    # Build name → id from added_tokens_decoder
    name_to_id = {}
    for k, v in added_tokens.items():
        content = v.get("content", "")
        if content:
            name_to_id[content] = int(k)

    # Extract key sentinels used by prompt builder
    for s in ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>",
              "<|ref_text|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        if s in name_to_id:
            result[s] = name_to_id[s]

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Higgs TTS to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model dir (e.g. bosonai--higgs-audio-v3-tts-4b)")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    model_dir = Path(args.input)
    if not model_dir.is_dir():
        sys.exit(f"not a directory: {model_dir}")

    # ── Config ───────────────────────────────────────────────────────────
    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    text_cfg = cfg.get("text_config", {})
    audio_cfg = cfg.get("audio_encoder_config", {})

    print(f"\nHiggs TTS v3 — {text_cfg.get('model_type', '?')}  "
          f"{text_cfg.get('num_hidden_layers')}L  "
          f"hidden={text_cfg.get('hidden_size')}  "
          f"heads={text_cfg.get('num_attention_heads')}/{text_cfg.get('num_key_value_heads')}")
    print(f"  Codec:        {audio_cfg.get('num_codebooks')} codebooks  "
          f"vocab_size={audio_cfg.get('vocab_size')}  "
          f"out_dim={audio_cfg.get('out_dim')}")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tok_json = model_dir / "tokenizer.json"
    tok_cfg_json = model_dir / "tokenizer_config.json"

    toks, merges = [], []
    special_ids = {}
    if tok_json.exists():
        toks, merges = extract_tokenizer_info(str(tok_json))
        print(f"  Tokens:        {len(toks)} entries from tokenizer.json")
        print(f"  Merges:        {len(merges)} entries from tokenizer.json")
    if tok_cfg_json.exists():
        special_ids = extract_special_token_ids(str(tok_cfg_json))
        print(f"  Specials:      {len(special_ids)} from tokenizer_config.json")

    # ── Safetensors ──────────────────────────────────────────────────────
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"no safetensors in {model_dir}")
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i
    print(f"  Safetensors:   {len(name_to_idx)} tensors in {len(st_files)} file(s)")

    # ── GGUF Writer ──────────────────────────────────────────────────────
    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    w = GGUFWriter(str(args.output), arch="higgs-tts", use_temp_file=False)

    # ── Metadata ─────────────────────────────────────────────────────────
    w.add_string("higgs.name", "HIGGS-audio-v3-TTS-4B")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    # Backbone (text_config)
    u32("higgs.text.n_layers",        text_cfg.get("num_hidden_layers", 36))
    u32("higgs.text.d_model",         text_cfg.get("hidden_size", 2560))
    u32("higgs.text.head_dim",        text_cfg.get("head_dim", 128))
    u32("higgs.text.n_heads",         text_cfg.get("num_attention_heads", 32))
    u32("higgs.text.n_kv_heads",      text_cfg.get("num_key_value_heads", 8))
    u32("higgs.text.ff_dim",          text_cfg.get("intermediate_size", 9728))
    u32("higgs.text.vocab_size",      text_cfg.get("vocab_size", 151936))
    u32("higgs.text.max_pos",         text_cfg.get("max_position_embeddings", 32768))
    f32("higgs.text.rope_theta",      text_cfg.get("rope_theta", 1_000_000))
    f32("higgs.text.rms_norm_eps",   text_cfg.get("rms_norm_eps", 1e-6))

    # Codec (audio_encoder_config)
    u32("higgs.codec.n_codebooks",    audio_cfg.get("num_codebooks", 8))
    u32("higgs.codec.vocab_size",     audio_cfg.get("vocab_size", 1026))
    u32("higgs.codec.out_dim",        audio_cfg.get("out_dim", 2560))
    u32("higgs.codec.sample_rate",    24000)
    u32("higgs.codec.hop_length",     320)

    # Global sentinels
    u32("higgs.audio_token_id",       cfg.get("audio_token_id", -100) & 0xFFFFFFFF)
    u32("higgs.ignore_index",         cfg.get("ignore_index", -100) & 0xFFFFFFFF)

    # Special token IDs for prompt builder
    for name, tid in special_ids.items():
        key = name.replace("<|", "").replace("|>", "").replace(":", "_")
        u32(f"higgs.token.{key}", tid)

    # BPE tokenizer
    if toks:
        w.add_token_list(toks)
    if merges:
        w.add_token_merges(merges)

    # ── Tensors ──────────────────────────────────────────────────────────
    n_mapped = 0
    n_skipped = 0
    n_convt = 0
    skipped_names = []

    for hf_name in sorted(name_to_idx.keys()):
        gn = map_tensor_name(hf_name)

        if gn is None:
            n_skipped += 1
            continue

        if gn.startswith("__UNMAPPED__:"):
            n_skipped += 1
            if len(skipped_names) < 30:
                skipped_names.append(hf_name)
            continue

        t = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()
        shape = t.shape

        # Determine dtype and transformation
        if is_snake_alpha(gn, shape):
            # Snake1d alpha [1, C, 1] → [C] F32
            t = transform_snake_alpha(t)
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        elif is_convt_weight(gn, shape):
            # ConvTranspose1d → wperm → F32
            t = transform_convt_wperm(t)
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
            n_convt += 1
        elif is_conv1d_weight(gn, shape):
            # Conv1d → permute → F16
            t = transform_conv1d_to_ggml(t)
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        elif shape == () or len(shape) == 0:
            # Scalar → F32
            t = np.array([t.item()], dtype=np.float32)
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        elif len(shape) == 1:
            # 1D (norm/bias/cluster_size) → F32
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            # 2D+ (Linear, Embedding) → F16
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)

        n_mapped += 1
        if n_mapped <= 30 or n_mapped % 100 == 0:
            print(f"  [{n_mapped}] {gn:55s} {t.shape}  {t.dtype}")

    if skipped_names:
        print(f"\n  WARNING: {len(skipped_names)} unmapped tensor(s):")
        for nm in skipped_names[:10]:
            print(f"    {nm}")

    print(f"\nMapped: {n_mapped}, skipped: {n_skipped}, conv_t wperm: {n_convt}")
    print(f"Writing {args.output}…")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = Path(args.output).stat().st_size / 1e9
    print(f"Done: {args.output}  ({sz:.2f} GB, {n_mapped} tensors)")


if __name__ == "__main__":
    import torch
    main()
