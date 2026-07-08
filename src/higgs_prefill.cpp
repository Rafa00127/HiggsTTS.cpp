// higgs_prefill_test_main.cpp — tokenize + encode_reference + delay_pattern → inputs_embeds

#include "higgs_prefill.h"
#include "higgs_tts.h"
#include "higgs_rvq_encode.h"
#include "core/bpe.h"
#include "core/audio_resample.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

static const int AUDIO_PLACEHOLDER_ID = -100;
static const int BOC_ID = 1024;
static const int EOC_ID = 1025;

// ── _to_mono_3d ──────────────────────────────────────────────────────────────
// Py: ensure waveform is [1, 1, T] torch float32 mono.
// C++: read_wav_mono_24k already returns mono float, this just documents the contract.
// Returns (samples, is_valid).
struct WavMono3D
{
    std::vector<float> samples; // flat float32, T samples
    int T = 0;
    int sample_rate = 0;
    bool ok = false;
};

static WavMono3D to_mono_3d(const std::vector<float> &raw, int raw_sr)
{
    WavMono3D out;
    out.samples = raw;
    out.T = (int)raw.size();
    out.sample_rate = raw_sr;
    out.ok = (out.T > 0);

    // Py同款非0项统计
    int nz = 0;
    float vmin = 0.0f, vmax = 0.0f;
    for (int i = 0; i < out.T; i++)
    {
        float v = raw[i];
        if (v != 0.0f)
        {
            if (nz == 0)
            {
                vmin = vmax = v;
            }
            else
            {
                if (v < vmin)
                    vmin = v;
                if (v > vmax)
                    vmax = v;
            }
            nz++;
        }
    }
    std::printf("mono3d 非0: %d/%d (%.1f%%), range=[%.6f,%.6f]\n",
                nz, out.T, 100.0f * nz / out.T, vmin, vmax);
    return out;
}

// ── prep_semantic_input: resample 24k→16k + squeeze + pad(160,160) ────────────
// Input:  wav.samples [T_24k] float, 24kHz mono
// Output: [T_16k + 320] float, 16kHz, padded both sides
// CPU only — preprocessing, no GPU needed
static std::vector<float> prep_semantic_input(const WavMono3D &wav)
{
    // Polyphase Kaiser-windowed sinc resample 24k → 16k (matches torchaudio/librosa)
    std::vector<float> resampled = core_audio::resample_polyphase(
        wav.samples.data(), wav.T, 24000, 16000);
    // pad(160, 160)
    size_t T_out = resampled.size();
    std::vector<float> out(T_out + 320);
    std::fill(out.begin(), out.begin() + 160, 0.0f);
    std::fill(out.end() - 160, out.end(), 0.0f);
    std::memcpy(out.data() + 160, resampled.data(), T_out * sizeof(float));
    return out;
}

// ── apply_delay_pattern ───────────────────────────────────────────────────────
// codes_TN flat [T*N], output flat [(T+N-1)*N]
// column c shifted down by c, BOC above, EOC below
static std::vector<int32_t> apply_delay_pattern(const std::vector<int32_t> &codes, int T, int N)
{
    int L = T + N - 1;
    std::vector<int32_t> out(L * N, EOC_ID);
    for (int c = 0; c < N; c++)
    {
        for (int t = 0; t < c; t++)
            out[t * N + c] = BOC_ID;
        for (int t = 0; t < T; t++)
            out[(c + t) * N + c] = codes[t * N + c];
    }
    return out;
}

// // ── fused_embed (CPU) ──────────────────────────────────────────────────────────
// // codes [N], weight [V*N, D] row-major → sum over codebooks → [D]
// static std::vector<float> fused_embed_cpu(const int32_t *codes, int N, int V,
//                                           const float *weight, int D)
// {
//     std::vector<float> result(D, 0.0f);
//     for (int c = 0; c < N; c++)
//     {
//         int row = c * V + codes[c];
//         const float *src = weight + row * D;
//         for (int j = 0; j < D; j++)
//             result[j] += src[j];
//     }
//     return result;
// }

// ── Hubert Feature Extractor (7×Conv1d+GroupNorm+GELU) ──────────────────────
// Input:  x [T_in, 1]  (ne0=T_in, ne1=1)
// Output: [T_out, 512]  (ne0=T_out, ne1=512)
static ggml_tensor *build_feature_extractor(ggml_context *ctx, ggml_tensor *x,
                                            higgs_test_model *m, int n_layers = 7)
{
    // strides/kernel per layer: [5,2,2,2,2,2,2], [10,3,3,3,3,2,2]
    // pad=0 all layers, bias=False all layers
    for (int i = 0; i < n_layers; i++)
    {
        auto &fe = m->sem.fe[i];
        int K = (int)fe.conv_w->ne[0];
        int IC = (int)fe.conv_w->ne[1];
        int OC = (int)fe.conv_w->ne[2];
        int s = (i == 0) ? 5 : 2;

        ggml_tensor *cols = ggml_im2col_rafa(ctx, x, K, s, 0, 1, fe.conv_w->type);
        ggml_tensor *w2d = ggml_reshape_2d(ctx, fe.conv_w, IC * K, OC);
        x = ggml_mul_mat(ctx, cols, w2d); // [T_out, OC]

        // GroupNorm only on layer 0
        if (fe.norm_w)
        {
            x = ggml_norm(ctx, x, 1e-5f);
            ggml_tensor *nw = ggml_reshape_2d(ctx, fe.norm_w, 1, OC);
            ggml_tensor *nb = ggml_reshape_2d(ctx, fe.norm_b, 1, OC);
            x = ggml_mul(ctx, x, nw);
            x = ggml_add(ctx, x, nb);
        }

        // GELU on all 7 layers
        x = ggml_gelu(ctx, x);

        // Transpose [T_out, OC] → [OC, T_out] for next im2col (skip after last)
        if (i < n_layers - 1)
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
    }
    return x; // [T_out, 512] from last layer
}

// ── Acoustic Encoder (DAC): conv1 + 5 blocks(ResUnits×3→Snake→Conv1d↓) + snake + conv2 ─
// Internal format: [T, C] (ne0=T, ne1=C). Snake_1d and bias operate on ne1=channel.
// Output: [T=379, C=256]
static ggml_tensor* build_acoustic_encoder(ggml_context* ctx, ggml_tensor* pcm, higgs_test_model* m) {
    ggml_tensor* x = pcm; // [1, T_audio]

    // conv1: Conv1d(1, 64, K=7, s=1, p=3, bias=True)
    {
        auto& w = m->ac_enc_conv1_w; auto& b = m->ac_enc_conv1_b;
        int K = (int)w->ne[0], IC = (int)w->ne[1], OC = (int)w->ne[2];
        auto cols = ggml_im2col_rafa(ctx, x, K, 1, (K-1)/2, 1, w->type);
        auto w2d = ggml_reshape_2d(ctx, w, K*IC, OC);
        x = ggml_mul_mat(ctx, w2d, cols);
        x = ggml_add(ctx, x, b);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));   // [T, OC] for snake
    }
    // 5 blocks
    for (int bi = 0; bi < 5; bi++) {
        auto& blk = m->ac_enc_blocks[bi];

        // ResUnits ×3: Snake1→Conv1d(K=7,dil)→Snake2→Conv1d(1×1)+skip
        int dils[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++) {
            auto& ru = blk.ru[r];
            int dil = dils[r];
            ggml_tensor* residual = x;                              // [T, C]
            ggml_tensor* h = ggml_snake_1d(ctx, x, ru.snake1_alpha);// [T, C]
            h = ggml_cont(ctx, ggml_transpose(ctx, h));             // [C, T] for im2col
            auto w = ru.conv1_w;
            int K = (int)w->ne[0], IC = (int)w->ne[1], OC = (int)w->ne[2];
            int pad = ((K-1)/2) * dil;
            auto cols = ggml_im2col_rafa(ctx, h, K, 1, pad, dil, w->type);
            auto w2d = ggml_reshape_2d(ctx, w, K*IC, OC);
            h = ggml_mul_mat(ctx, w2d, cols);
            h = ggml_add(ctx, h, ru.conv1_b);
            h = ggml_cont(ctx, ggml_transpose(ctx, h));            // [T,OC] for snake
            h = ggml_snake_1d(ctx, h, ru.snake2_alpha);            // [T, C]
            h = ggml_cont(ctx, ggml_transpose(ctx, h));             // [C, T] for 1x1 conv
            auto w1 = ru.conv2_w;
            IC = (int)w1->ne[1]; OC = (int)w1->ne[2];
            auto w1_2d = ggml_reshape_2d(ctx, w1, IC, OC);
            if (w1_2d->type != GGML_TYPE_F32)
                w1_2d = ggml_cast(ctx, w1_2d, GGML_TYPE_F32);
            h = ggml_mul_mat(ctx, w1_2d, h);
            h = ggml_add(ctx, h, ru.conv2_b);
            h = ggml_cont(ctx, ggml_transpose(ctx, h));
            x = ggml_add(ctx, residual, h);                         // [T, C]
        }
        // Snake1 then downsample Conv1d (pad from Python model)
        int blk_pads[5] = {4, 3, 2, 1, 2};
        int blk_strides[5] = {8, 5, 4, 2, 3};
        x = ggml_snake_1d(ctx, x, blk.snake1_alpha);               // [T, C]
        x = ggml_cont(ctx, ggml_transpose(ctx, x));                 // [C, T]
        auto w = blk.conv1_w;
        int K = (int)w->ne[0], IC = (int)w->ne[1], OC = (int)w->ne[2];
        int s = blk_strides[bi];
        int p = blk_pads[bi];
        auto cols = ggml_im2col_rafa(ctx, x, K, s, p, 1, w->type);
        auto w2d = ggml_reshape_2d(ctx, w, K*IC, OC);
        x = ggml_mul_mat(ctx, w2d, cols);
        x = ggml_add(ctx, x, blk.conv1_b);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
    }
    // Final snake + conv2
    x = ggml_snake_1d(ctx, x, m->ac_enc_snake1);                   // [T, 2048]
    x = ggml_cont(ctx, ggml_transpose(ctx, x));                     // [2048, T]
    {
        auto& w = m->ac_enc_conv2_w; auto& b = m->ac_enc_conv2_b;
        int K = (int)w->ne[0], IC = (int)w->ne[1], OC = (int)w->ne[2];
        auto cols = ggml_im2col_rafa(ctx, x, K, 1, (K-1)/2, 1, w->type);
        auto w2d = ggml_reshape_2d(ctx, w, K*IC, OC);
        x = ggml_mul_mat(ctx, w2d, cols);
        x = ggml_add(ctx, x, b);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
    }
    return x;  // [T, 256]
}

// ── Semantic Encoder: conv → 2×Block(2×ResUnit→Conv1d) ──────────────────────
// Input:  se_in [T, 1024]  (ne0=T, ne1=1024)
// Output: [1024, T_enc]    (ne0=1024, ne1=T_enc)
static ggml_tensor *build_semantic_encoder(ggml_context *ctx, ggml_tensor *se_in,
                                           higgs_test_model *m)
{
    // First conv: Conv1d(1024, 1024, K=3, s=1, bias=False)
    ggml_tensor *x = se_in; // [T, 1024]
    {
        ggml_tensor *w = m->enc_sem_conv_w; // [K, IC, OC] ne0=K
        int K = (int)w->ne[0];
        int IC = (int)w->ne[1];
        int OC = (int)w->ne[2];
        ggml_tensor *cols = ggml_im2col_rafa(ctx, x, K, 1, (K - 1) / 2, 1, w->type);
        ggml_tensor *w2d = ggml_reshape_2d(ctx, w, K * IC, OC);
        x = ggml_mul_mat(ctx, w2d, cols);                     // [OC, T]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [T, OC]
    }

    for (int bi = 0; bi < 2; bi++)
    {
        auto &blk = m->enc_sem_blocks[bi];

        // ResUnits: ELU → Conv1d(dilated, bias=False) → ELU → Conv1d(1×1, bias=False) → +residual
        for (auto &ru : blk.ru)
        {
            ggml_tensor *residual = x;
            ggml_tensor *h = ggml_elu(ctx, x);
            // Conv1d dilated
            {
                ggml_tensor *w = ru.conv1_w; // [K, IC, OC]
                int K = (int)w->ne[0];
                int IC = (int)w->ne[1];
                int OC = (int)w->ne[2];
                // dilation from weight comment: conv1 usually dil=1 for semantic encoder
                int dil = 1;
                int pad = ((K - 1) / 2) * dil;
                ggml_tensor *cols = ggml_im2col_rafa(ctx, h, K, 1, pad, dil, w->type);
                ggml_tensor *w2d = ggml_reshape_2d(ctx, w, K * IC, OC);
                h = ggml_mul_mat(ctx, w2d, cols);                     // [OC, T]
                h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 0, 2, 3)); // [T, OC]
            }
            h = ggml_elu(ctx, h);
            // Conv1d 1×1 (stored as 2D weight)
            {
                ggml_tensor *w = ru.conv2_w; // [1, IC, OC] → reshape to [IC, OC]
                int IC = (int)w->ne[1];
                int OC = (int)w->ne[2];
                ggml_tensor *w2d = ggml_reshape_2d(ctx, w, IC, OC);
                // x is [T, IC], need [IC, T] for mul_mat
                ggml_tensor *ht = ggml_cont(ctx, ggml_permute(ctx, h, 1, 0, 2, 3)); // [IC, T]
                ggml_tensor *out = ggml_mul_mat(ctx, w2d, ht);                      // [OC, T]
                out = ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));           // [T, OC]
                h = out;
            }
            x = ggml_add(ctx, residual, h);
        }

        // Block conv: ELU → Conv1d(s=1)
        {
            x = ggml_elu(ctx, x);
            ggml_tensor *w = blk.conv_w; // [K, IC, OC]
            int K = (int)w->ne[0];
            int IC = (int)w->ne[1];
            int OC = (int)w->ne[2];
            ggml_tensor *cols = ggml_im2col_rafa(ctx, x, K, 1, (K - 1) / 2, 1, w->type);
            ggml_tensor *w2d = ggml_reshape_2d(ctx, w, K * IC, OC);
            x = ggml_add(ctx, ggml_mul_mat(ctx, w2d, cols), blk.conv_b); // [OC, T]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));        // [T, OC]
        }
    }
    return ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [OC, T]
}

// 下面的函数大概率有些地方是不对的
// ── transformer_layer (placeholder, not used in this test) ────────────────────
static ggml_tensor *transformer_layer(ggml_context *ctx, ggml_tensor *x,
                                      higgs_test_model *m, int layer_idx, ggml_tensor *pos, ggml_tensor *mask,
                                      int hd, int nh, int nkv, int nt, ggml_tensor *&out_attn)
{
    auto &L = m->layer[layer_idx];
    ggml_tensor *norm = ggml_rms_norm(ctx, x, 1e-6f);
    norm = ggml_mul(ctx, norm, L.attn_norm);
    ggml_tensor *Q = ggml_mul_mat(ctx, L.attn_q, norm);
    ggml_tensor *K = ggml_mul_mat(ctx, L.attn_k, norm);
    ggml_tensor *V = ggml_mul_mat(ctx, L.attn_v, norm);
    Q = ggml_reshape_3d(ctx, Q, hd, nh, nt);
    K = ggml_reshape_3d(ctx, K, hd, nkv, nt);
    V = ggml_reshape_3d(ctx, V, hd, nkv, nt);
    Q = ggml_rope_ext(ctx, Q, pos, nullptr, hd, 2, 8192, 1e6f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, pos, nullptr, hd, 2, 8192, 1e6f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
    ggml_tensor *attn = ggml_flash_attn_ext(ctx, Q, K, V, mask, 1.0f / sqrtf((float)hd), 0.0f, 0.0f);
    attn = ggml_cont(ctx, attn);
    attn = ggml_reshape_2d(ctx, attn, hd * nh, nt);
    out_attn = ggml_mul_mat(ctx, L.attn_o, attn);
    ggml_tensor *out = ggml_add(ctx, x, out_attn);
    ggml_tensor *ffn = ggml_rms_norm(ctx, out, 1e-6f);
    ffn = ggml_mul(ctx, ffn, L.ffn_norm);
    ggml_tensor *gate = ggml_mul_mat(ctx, L.ffn_gate, ffn);
    ggml_tensor *up = ggml_mul_mat(ctx, L.ffn_up, ffn);
    gate = ggml_silu(ctx, gate);
    ffn = ggml_mul(ctx, gate, up);
    ffn = ggml_mul_mat(ctx, L.ffn_down, ffn);
    out = ggml_add(ctx, out, ffn);
    return out;
}

bool higgs_prefill_encode(higgs_test_model* m, const float* audio, int n_samples,
                           std::vector<int32_t>& codes, int& T_frames) {
    if (!m || !audio || n_samples < 24000) return false;

    WavMono3D wav3d;
    wav3d.samples.assign(audio, audio + n_samples);
    wav3d.T = n_samples;
    wav3d.sample_rate = 24000;
    wav3d.ok = true;
    std::vector<float> sem_input = prep_semantic_input(wav3d);

    ggml_init_params ip = { m->compute_meta.size(), m->compute_meta.data(), true };
    ggml_context *ctx = ggml_init(ip);
    if (!ctx) return false;

    int T_in = (int)sem_input.size();
    ggml_tensor *in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T_in);
    ggml_set_input(in);
    ggml_tensor *out = build_feature_extractor(ctx, in, m, 7);

    // FP: transpose → LN → Linear(512→768)
    ggml_tensor *out_t = ggml_cont(ctx, ggml_transpose(ctx, out));
    ggml_tensor *out_norm = ggml_norm(ctx, out_t, 1e-5f);
    ggml_tensor *ln_w = ggml_repeat(ctx, m->sem.fp_ln_w, out_norm);
    ggml_tensor *ln_b = ggml_repeat(ctx, m->sem.fp_ln_b, out_norm);
    out_norm = ggml_mul(ctx, ln_w, out_norm);
    out_norm = ggml_add(ctx, ln_b, out_norm);
    ggml_tensor *fp_out = ggml_mul_mat(ctx, m->sem.fp_w, out_norm);
    fp_out = ggml_add(ctx, ggml_repeat(ctx, m->sem.fp_b, fp_out), fp_out);

        int K_pce = 128, Cg_pce = 48, Ng = 16;
        {
            // Pre-fused weight from model loading: [K*Cg, OC] = [6144, 768]
            ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K_pce * Cg_pce, 768);
            ggml_set_input(w);

            // Grouped Conv1d: 16 groups. fp_out [768,379] C-fast T-slow.
            int T = (int)fp_out->ne[1];
            size_t nb1_w = K_pce * Cg_pce * (int64_t)sizeof(float);
            // 1. fp_out [768,379] → transpose + cont → [379,768] (T-fast C-slow)
            ggml_tensor *x_t = ggml_cont(ctx, ggml_transpose(ctx, fp_out));
            std::vector<ggml_tensor *> g_outs(Ng);
            for (int g = 0; g < Ng; g++)
            {
                // 2. view group g: Cg*T elements → [379, Cg]
                ggml_tensor *xg_t = ggml_view_2d(ctx, x_t, T, Cg_pce, x_t->nb[1],
                                                 g * Cg_pce * x_t->nb[1]);
                // 3. transpose + cont → [Cg, T] for im2col
                ggml_tensor *xg = ggml_cont(ctx, ggml_transpose(ctx, xg_t));
                // im2col + matmul
                ggml_tensor *cols_full = ggml_im2col_rafa(ctx, xg, K_pce, 1, 64, 1, GGML_TYPE_F32);
                ggml_tensor *cols = ggml_view_2d(ctx, cols_full, K_pce * Cg_pce, T,
                                                 cols_full->nb[1], 0);
                ggml_tensor *wg = ggml_view_2d(ctx, w, K_pce * Cg_pce, Cg_pce, nb1_w,
                                               g * Cg_pce * nb1_w);
                g_outs[g] = ggml_mul_mat(ctx, cols, wg); // [T, Cg]
            }
            // Concat 16 × [T,Cg] → [T,768] along ne1
            ggml_tensor *pos_emb = g_outs[0];
            for (int g = 1; g < Ng; g++)
                pos_emb = ggml_concat(ctx, pos_emb, g_outs[g], 1);
            // [T, 768]

            // Bias along ne1=768
            if (m->sem.pce_bias)
            {
                ggml_tensor *pb = ggml_repeat(ctx,
                                              ggml_reshape_2d(ctx, m->sem.pce_bias, 1, 768), pos_emb);
                pos_emb = ggml_add(ctx, pb, pos_emb);
            }

            // GELU
            pos_emb = ggml_gelu(ctx, pos_emb);

            // Transpose → [768, T] matching py [1,T,C]
            pos_emb = ggml_cont(ctx, ggml_transpose(ctx, pos_emb));

            // Residual + LayerNorm
            ggml_tensor *x = ggml_add(ctx, fp_out, pos_emb);
            x = ggml_norm(ctx, x, 1e-5f);
            if (m->sem.post_ln_w && m->sem.post_ln_b)
            {
                ggml_tensor *ln_w = ggml_repeat(ctx, m->sem.post_ln_w, x);
                ggml_tensor *ln_b = ggml_repeat(ctx, m->sem.post_ln_b, x);
                x = ggml_mul(ctx, ln_w, x);
                x = ggml_add(ctx, ln_b, x);
            }

            // WavLM encoder: 12 layers, collect all outputs for mean
            const int hd = 64, nh = 12, nkv = 12;
            const int n_enc_layers = (int)m->sem.layers.size();
            int T_enc = (int)x->ne[1];

            auto proj = [&](ggml_tensor *w, ggml_tensor *b, ggml_tensor *src)
            {
                ggml_tensor *y = ggml_mul_mat(ctx, w, src);
                y = ggml_add(ctx, y, b);
                return y;
            };

            ggml_tensor *enc_x = x; // [768, T]
            std::vector<ggml_tensor *> layer_outputs;
            for (int li = 0; li < n_enc_layers; li++)
            {
                auto &L = m->sem.layers[li];

                // Q/K/V projections
                ggml_tensor *Q = proj(L.attn_q_w, L.attn_q_b, enc_x);
                ggml_tensor *K = proj(L.attn_k_w, L.attn_k_b, enc_x);
                ggml_tensor *V = proj(L.attn_v_w, L.attn_v_b, enc_x);

                // Reshape + permute: [768, T] → [hd, nh, T] → [hd, T, nh]
                Q = ggml_reshape_3d(ctx, Q, hd, nh, T_enc);
                Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
                K = ggml_reshape_3d(ctx, K, hd, nkv, T_enc);
                K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
                V = ggml_reshape_3d(ctx, V, hd, nkv, T_enc);
                V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

                // Flash attn
                ggml_tensor *attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr,
                                                        1.0f / sqrtf((float)hd), 0.0f, 0.0f);
                attn = ggml_cont(ctx, attn);
                attn = ggml_reshape_2d(ctx, attn, hd * nh, T_enc);

                // O projection
                ggml_tensor *attn_out = ggml_mul_mat(ctx, L.attn_out_w, attn);
                attn_out = ggml_add(ctx, attn_out, L.attn_out_b);

                // POST-NORM: residual + LN (attn)
                ggml_tensor *a_residual = enc_x;
                ggml_tensor *h = ggml_add(ctx, a_residual, attn_out);
                h = ggml_norm(ctx, h, 1e-5f);
                if (L.ln_w)
                {
                    ggml_tensor *lw = ggml_repeat(ctx, L.ln_w, h);
                    ggml_tensor *lb = ggml_repeat(ctx, L.ln_b, h);
                    h = ggml_mul(ctx, lw, h);
                    h = ggml_add(ctx, lb, h);
                }

                // FFN: GELU(ffn1) @ ffn2
                ggml_tensor *ffn = ggml_mul_mat(ctx, L.ffn1_w, h);
                ffn = ggml_add(ctx, ffn, L.ffn1_b);
                ffn = ggml_gelu(ctx, ffn);
                ffn = ggml_mul_mat(ctx, L.ffn2_w, ffn);
                ffn = ggml_add(ctx, ffn, L.ffn2_b);

                // POST-NORM: residual + LN (ffn)
                enc_x = ggml_add(ctx, h, ffn);
                enc_x = ggml_norm(ctx, enc_x, 1e-5f);
                if (L.fin_ln_w)
                {
                    ggml_tensor *lw = ggml_repeat(ctx, L.fin_ln_w, enc_x);
                    ggml_tensor *lb = ggml_repeat(ctx, L.fin_ln_b, enc_x);
                    enc_x = ggml_mul(ctx, lw, enc_x);
                    enc_x = ggml_add(ctx, lb, enc_x);
                }
                layer_outputs.push_back(enc_x);
            }

            // Average all 12 layer outputs → wavlm_hidden [768, T]
            ggml_tensor *wavlm_hidden = layer_outputs[0];
            for (int li = 1; li < n_enc_layers; li++)
                wavlm_hidden = ggml_add(ctx, wavlm_hidden, layer_outputs[li]);
            wavlm_hidden = ggml_scale(ctx, wavlm_hidden, 1.0f / n_enc_layers);

            // ── Semantic Encoder: conv → 2×Block(2×ResUnit→Conv1d) ────────────
            // Input: [C=768, T], output: [C=768, T]
            ggml_tensor *se_x = wavlm_hidden; // [768, T]
#if 0
            // Full Semantic Encoder: 2 blocks, each 2 ResUnits + 1 Conv1d
            auto res_unit = [&](ggml_tensor*& x, higgs_enc_sem_resunit& ru, int dil) {
                ggml_tensor* residual = x;
                ggml_tensor* h = ggml_elu(ctx, x);
                ggml_tensor* w = ru.conv1_w;
                int K = (int)w->ne[0], IC = (int)w->ne[1], OC = (int)w->ne[2];
                int pad = ((K-1)/2)*dil;
                ggml_tensor* cols = ggml_im2col_rafa(ctx, h, K, 1, pad, dil, w->type);
                ggml_tensor* w2d = ggml_reshape_2d(ctx, w, K*IC, OC);
                h = ggml_mul_mat(ctx, cols, w2d);
                h = ggml_cont(ctx, ggml_transpose(ctx, h));
                h = ggml_elu(ctx, h);
                ggml_tensor* w1 = ru.conv2_w;
                IC = (int)w1->ne[1]; OC = (int)w1->ne[2];
                ggml_tensor* w1_2d = ggml_reshape_2d(ctx, w1, IC, OC);
                if (w1_2d->type != GGML_TYPE_F32)
                    w1_2d = ggml_cpy(ctx, w1_2d, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, OC));
                h = ggml_mul_mat(ctx, h, w1_2d);
                h = ggml_cont(ctx, ggml_transpose(ctx, h));
                x = ggml_add(ctx, residual, h);
            };
            for (int bi = 0; bi < 2; bi++) {
                auto& blk = m->enc_sem_blocks[bi];
                for (auto& ru : blk.ru)
                    res_unit(se_x, ru, 1);  // dil=1
                // Block conv
                ggml_tensor* w = blk.conv_w;
                int K = (int)w->ne[0], IC = (int)w->ne[1], OC = (int)w->ne[2];
                ggml_tensor* cols = ggml_im2col_rafa(ctx, se_x, K, 1, (K-1)/2, 1, w->type);
                ggml_tensor* w2d = ggml_reshape_2d(ctx, w, K*IC, OC);
                se_x = ggml_mul_mat(ctx, cols, w2d);
                se_x = ggml_cont(ctx, ggml_transpose(ctx, se_x));
                se_x = ggml_add(ctx, se_x, blk.conv_b);
            }
#endif
            // Full Semantic Encoder: conv + 2 blocks
            // First conv
            {
                auto &w = m->enc_sem_conv_w;
                int K = (int)w->ne[0], IC = (int)w->ne[1], OC = (int)w->ne[2];
                auto cols = ggml_im2col_rafa(ctx, se_x, K, 1, (K - 1) / 2, 1, w->type);
                auto w2d = ggml_reshape_2d(ctx, w, K * IC, OC);
                se_x = ggml_mul_mat(ctx, cols, w2d);
                se_x = ggml_cont(ctx, ggml_transpose(ctx, se_x));
            }
            for (int bi = 0; bi < 2; bi++)
            {
                auto &blk = m->enc_sem_blocks[bi];
                for (auto &ru : blk.ru)
                {
                    ggml_tensor *residual = se_x;
                    ggml_tensor *h = ggml_elu(ctx, se_x);
                    auto w = ru.conv1_w;
                    int K = (int)w->ne[0], IC = (int)w->ne[1], OC = (int)w->ne[2];
                    int dil = 1, pad = ((K - 1) / 2) * dil;
                    auto cols = ggml_im2col_rafa(ctx, h, K, 1, pad, dil, w->type);
                    auto w2d = ggml_reshape_2d(ctx, w, K * IC, OC);
                    h = ggml_mul_mat(ctx, cols, w2d);
                    h = ggml_cont(ctx, ggml_transpose(ctx, h));
                    h = ggml_elu(ctx, h);
                    auto w1 = ru.conv2_w;
                    IC = (int)w1->ne[1];
                    OC = (int)w1->ne[2];
                    auto w1_2d = ggml_reshape_2d(ctx, w1, IC, OC);
                    if (w1_2d->type != GGML_TYPE_F32)
                        w1_2d = ggml_cpy(ctx, w1_2d, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, OC));
                    h = ggml_mul_mat(ctx, h, w1_2d);
                    h = ggml_cont(ctx, ggml_transpose(ctx, h));
                    se_x = ggml_add(ctx, residual, h);
                }
                auto w = blk.conv_w;
                int K = (int)w->ne[0], IC = (int)w->ne[1], OC = (int)w->ne[2];
                auto cols = ggml_im2col_rafa(ctx, se_x, K, 1, (K - 1) / 2, 1, w->type);
                auto w2d = ggml_reshape_2d(ctx, w, K * IC, OC);
                se_x = ggml_mul_mat(ctx, cols, w2d);
                se_x = ggml_cont(ctx, ggml_transpose(ctx, se_x));
                se_x = ggml_add(ctx, se_x, blk.conv_b);
            }
            // Output: [T, C]
            ggml_tensor *sem_enc_out = ggml_cont(ctx, ggml_transpose(ctx, se_x));
            // ggml_set_output(sem_enc_out);

            // ── Acoustic Encoder ────────────────────────────────────────────
            // Predict T_ac from conv chain (like _get_conv1d_output_lengths)
            int T_ac_pred = wav3d.T;
            // conv1: K=7, s=1, p=3
            T_ac_pred = (T_ac_pred + 2*3 - 1*6 - 1) / 1 + 1;
            // 5 blocks
            int bK[5] = {16, 10, 8, 4, 6};
            int bs[5] = {8, 5, 4, 2, 3};
            int bp[5] = {4, 3, 2, 1, 2};
            for (int bi = 0; bi < 5; bi++) {
                // 3 ResUnits preserve length (p=dil*3, dil*(K-1)=dil*6, cancel out)
                // block conv
                T_ac_pred = (T_ac_pred + 2*bp[bi] - 1*(bK[bi]-1) - 1) / bs[bi] + 1;
            }
            // conv2: K=3, s=1, p=1
            T_ac_pred = (T_ac_pred + 2*1 - 1*2 - 1) / 1 + 1;

            int T_sem = (int)fp_out->ne[1];
            int hop = 8 * 5 * 4 * 2 * 3;  // = 960, total stride
            int pad_frames = T_sem - T_ac_pred;
            int pad_L = (pad_frames / 2) * hop;
            int pad_R = (pad_frames - pad_frames/2) * hop;
            std::vector<float> wav_pad(pad_L + wav3d.T + pad_R, 0.0f);
            std::memcpy(wav_pad.data() + pad_L, wav3d.samples.data(), wav3d.T * sizeof(float));

            ggml_tensor* pcm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, (int)wav_pad.size());
            ggml_set_input(pcm);
            ggml_tensor* ac_out = build_acoustic_encoder(ctx, pcm, m);

            // Concat + FC
            ggml_tensor* combined = ggml_concat(ctx, ac_out, sem_enc_out, 1); // [T, 1024]
            ggml_tensor* cmb_t = ggml_cont(ctx, ggml_transpose(ctx, combined));// [1024, T]
            ggml_tensor* fused = ggml_mul_mat(ctx, m->fc_w, cmb_t);            // [1024, T]
            fused = ggml_add(ctx, fused, m->fc_b);
            fused = ggml_cont(ctx, ggml_transpose(ctx, fused));                // [T, 1024]

            ggml_cgraph *gf = ggml_new_graph_custom(ctx, 16384, false);
            ggml_build_forward_expand(gf, fused);

            ggml_backend_sched_reset(m->sched);
            if (!ggml_backend_sched_alloc_graph(m->sched, gf))
            {
                std::fprintf(stderr, "alloc fail\n");
                ggml_free(ctx);
                return 1;
            }
            // if (use_py_fp)
            //     ggml_backend_tensor_set(fp_out, fp_out_py_data.data(), 0, fp_out_py_data.size() * sizeof(float));
            // else
            ggml_backend_tensor_set(in, sem_input.data(), 0, T_in * sizeof(float));
            ggml_backend_tensor_set(w, m->sem.pce_weight_data.data(), 0,
                                     K_pce * Cg_pce * 768 * sizeof(float));
            ggml_backend_tensor_set(pcm, wav_pad.data(), 0, wav_pad.size() * sizeof(float));
            if (ggml_backend_sched_graph_compute(m->sched, gf) != GGML_STATUS_SUCCESS)
            {
                std::fprintf(stderr, "compute fail\n");
                ggml_free(ctx);
                return 1;
            }

            // RVQ Encode: 8-stage residual quantization on CPU
            {
                int T_f = (int)fused->ne[0], D = (int)fused->ne[1];       // [379, 1024]
                int cb_dim = (int)m->quant[0].codebook->ne[0];              // 64
                int cb_size = (int)m->quant[0].codebook->ne[1];             // 1024
                int N = 8;

                // Read fused [T_f, D] from GPU
                int n_fused = T_f * D;
                std::vector<float> residual(n_fused);
                ggml_backend_tensor_get(fused, residual.data(), 0, n_fused * sizeof(float));

                // Read all codebooks and projection weights from GPU (may be F16 or quantized)
                auto load_f32 = [&](ggml_tensor* t, std::vector<float>& dst) {
                    int n = (int)ggml_nelements(t);
                    dst.resize(n);
                    if (t->type == GGML_TYPE_F32) {
                        ggml_backend_tensor_get(t, dst.data(), 0, n * sizeof(float));
                    } else if (t->type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> raw(n);
                        ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_fp16_t));
                        ggml_fp16_to_fp32_row(raw.data(), dst.data(), n);
                    } else {
                        auto* traits = ggml_get_type_traits(t->type);
                        std::vector<uint8_t> raw(ggml_nbytes(t));
                        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
                        traits->to_float(raw.data(), dst.data(), n);
                    }
                };

                std::vector<float> cb_data[8], pi_w[8], pi_b[8], po_w[8], po_b[8];
                for (int q = 0; q < N; q++) {
                    load_f32(m->quant[q].codebook, cb_data[q]);
                    load_f32(m->quant[q].proj_in_w, pi_w[q]);
                    load_f32(m->quant[q].proj_in_b, pi_b[q]);
                    load_f32(m->quant[q].proj_out_w, po_w[q]);
                    load_f32(m->quant[q].proj_out_b, po_b[q]);
                }

                std::vector<int32_t> all_codes;
                for (int q = 0; q < N; q++) {
                    // proj_in: z = residual @ W.T + b  (1x1 Conv: C→cdim)
                    std::vector<float> z(T_f * cb_dim);
                    for (int t = 0; t < T_f; t++) {
                        for (int c = 0; c < cb_dim; c++) {
                            float s = pi_b[q][c];
                            for (int d = 0; d < D; d++)
                                s += residual[t + d * T_f] * pi_w[q][d + c * D];
                            z[t + c * T_f] = s;
                        }
                    }

                    // Transpose z to row-major [T, cdim] for nearest_neighbor
                    std::vector<float> z_rm(T_f * cb_dim);
                    for (int t = 0; t < T_f; t++)
                        for (int c = 0; c < cb_dim; c++)
                            z_rm[t * cb_dim + c] = z[t + c * T_f];

                    std::vector<int32_t> codes_q;
                    rvq_nearest_neighbor(z_rm.data(), T_f, cb_data[q].data(), cb_size, cb_dim, codes_q);
                    for (int t = 0; t < T_f; t++) all_codes.push_back(codes_q[t]);

                    // Residual -= proj_out(codebook[codes_q[t]]) + proj_out_b
                    for (int t = 0; t < T_f; t++) {
                        const float* cb_entry = cb_data[q].data() + (size_t)codes_q[t] * cb_dim;
                        for (int d = 0; d < D; d++) {
                            float s = po_b[q][d];
                            for (int c = 0; c < cb_dim; c++)
                                s += cb_entry[c] * po_w[q][c + d * cb_dim];
                            residual[t + d * T_f] -= s;
                        }
                    }
                }

                // all_codes is q-major, transpose to t-major
                codes.resize(N * T_f);
                for (int t = 0; t < T_f; t++)
                    for (int q = 0; q < N; q++)
                        codes[t * N + q] = all_codes[q * T_f + t];
                T_frames = T_f;
            }

            ggml_free(ctx);
            return true;
        }
}
