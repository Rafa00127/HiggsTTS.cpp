// higgs_tts.h — Higgs TTS model structs and loader
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct higgs_test_quant {
    ggml_tensor* proj_in_w = nullptr;      // [1088, 1088] RVQ input projection
    ggml_tensor* proj_in_b = nullptr;      // [1088]
    ggml_tensor* codebook = nullptr;       // [1024, 1088]
    ggml_tensor* proj_out_w = nullptr;     // [1088, 1088]
    ggml_tensor* proj_out_b = nullptr;     // [1088]
};

// ── Codec Encoder sub-structs ────────────────────────────────────────────────

// Acoustic encoder (DAC): conv1 + 5 blocks + conv2 + snake
struct higgs_ac_enc_resunit {
    ggml_tensor* conv1_w = nullptr;    // [7, C, C]
    ggml_tensor* conv1_b = nullptr;    // [C]
    ggml_tensor* conv2_w = nullptr;    // [1, C, C]
    ggml_tensor* conv2_b = nullptr;    // [C]
    ggml_tensor* snake1_alpha = nullptr;  // [C]
    ggml_tensor* snake2_alpha = nullptr;  // [C]
};
struct higgs_ac_enc_block {
    ggml_tensor* conv1_w = nullptr;    // Conv1d downsample [K, IC, OC]
    ggml_tensor* conv1_b = nullptr;    // [OC]
    ggml_tensor* snake1_alpha = nullptr;  // [OC]
    higgs_ac_enc_resunit ru[3];
};

// Semantic encoder (encoder_semantic): Conv1d + blocks with ResUnits
struct higgs_enc_sem_resunit {
    ggml_tensor* conv1_w = nullptr;         // Conv1d dilated, bias=False
    ggml_tensor* conv2_w = nullptr;         // Conv1d 1x1, bias=False
};
struct higgs_enc_sem_block {
    ggml_tensor* conv_w = nullptr;
    ggml_tensor* conv_b = nullptr;
    std::vector<higgs_enc_sem_resunit> ru;
};

// Hubert/WavLM semantic model
struct higgs_sem_layer {
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* ffn1_w = nullptr;
    ggml_tensor* ffn1_b = nullptr;
    ggml_tensor* ffn2_w = nullptr;
    ggml_tensor* ffn2_b = nullptr;
    ggml_tensor* ln_w = nullptr;           // input layernorm
    ggml_tensor* ln_b = nullptr;
    ggml_tensor* fin_ln_w = nullptr;        // final layernorm
    ggml_tensor* fin_ln_b = nullptr;
};
struct higgs_sem_model {
    struct FeLayer {
        ggml_tensor* conv_w = nullptr;
        ggml_tensor* norm_w = nullptr;
        ggml_tensor* norm_b = nullptr;
    };
    std::vector<FeLayer> fe;           // 7 feature extractor layers
    ggml_tensor* fp_w = nullptr;            // feature projection Linear
    ggml_tensor* fp_b = nullptr;
    ggml_tensor* fp_ln_w = nullptr;         // feature projection LayerNorm
    ggml_tensor* fp_ln_b = nullptr;
    std::vector<higgs_sem_layer> layers;    // encoder transformer layers
    // pos_conv_embed: grouped Conv1d(768,768,K=128,pad=64,groups=16) + weight_norm
    ggml_tensor* pce_orig0 = nullptr;       // weight_g  [128,1,1]
    ggml_tensor* pce_orig1 = nullptr;       // weight_v  [128,48,768]
    ggml_tensor* pce_bias   = nullptr;      // bias [768]
    std::vector<float> pce_weight_data;     // pre-fused weight [6144,768] = K*Cg, OC
    ggml_tensor* post_ln_w = nullptr;       // encoder.layer_norm
    ggml_tensor* post_ln_b = nullptr;
    ggml_tensor* top_ln_w = nullptr;        // model.layer_norm
    ggml_tensor* top_ln_b = nullptr;
};

struct higgs_test_model {
    int N = 8;           // codebooks
    int codec_dim = 1024;

    higgs_test_quant quant[8];

    // ── Codec Encoder ──
    ggml_tensor* ac_enc_conv1_w = nullptr;         // first conv [7, 1, 64]
    ggml_tensor* ac_enc_conv1_b = nullptr;         // [64]
    higgs_ac_enc_block ac_enc_blocks[5];           // 5 downsampling blocks
    ggml_tensor* ac_enc_conv2_w = nullptr;         // output conv [3, 2048, 256]
    ggml_tensor* ac_enc_conv2_b = nullptr;         // [256]
    ggml_tensor* ac_enc_snake1 = nullptr;          // output snake [2048]
    higgs_enc_sem_block enc_sem_blocks[2];  // semantic encoder (2 blocks)
    ggml_tensor* enc_sem_conv_w = nullptr;  //   first conv
    higgs_sem_model sem;                    // Hubert semantic model
    ggml_tensor* fc_w = nullptr;            // fusion fc
    ggml_tensor* fc_b = nullptr;
    ggml_tensor* fc1_w = nullptr;           // fusion fc1
    ggml_tensor* fc1_b = nullptr;

    ggml_tensor* fc2_w = nullptr;           // [256, 1024]
    ggml_tensor* fc2_b = nullptr;           // [256]

    ggml_tensor* conv1_w = nullptr;        // [7, 1024, 256]
    ggml_tensor* conv1_b = nullptr;        // [1024]

    ggml_tensor* snake1_alpha = nullptr;    // [1024] — block.0.snake1.alpha
    ggml_tensor* convt1_w = nullptr;         // [1024, 8192] w_perm [IC, K*OC]
    ggml_tensor* convt1_b = nullptr;         // [512]

    // ResUnit1 (dilation=1)
    ggml_tensor* ru1_s1_alpha = nullptr;      // [512] snake1.alpha
    ggml_tensor* ru1_c1_w = nullptr;          // [7, 512, 512] conv1
    ggml_tensor* ru1_c1_b = nullptr;          // [512]
    ggml_tensor* ru1_s2_alpha = nullptr;      // [512] snake2.alpha
    ggml_tensor* ru1_c2_w = nullptr;          // [1, 512, 512] conv2
    ggml_tensor* ru1_c2_b = nullptr;          // [512]

    // ResUnit2 (dilation=3)
    ggml_tensor* ru2_s1_alpha = nullptr;      // [512]
    ggml_tensor* ru2_c1_w = nullptr;          // [7, 512, 512]
    ggml_tensor* ru2_c1_b = nullptr;          // [512]
    ggml_tensor* ru2_s2_alpha = nullptr;      // [512]
    ggml_tensor* ru2_c2_w = nullptr;          // [1, 512, 512]
    ggml_tensor* ru2_c2_b = nullptr;          // [512]

    // ResUnit3 (dilation=9)
    ggml_tensor* ru3_s1_alpha = nullptr;      // [512]
    ggml_tensor* ru3_c1_w = nullptr;          // [7, 512, 512]
    ggml_tensor* ru3_c1_b = nullptr;          // [512]
    ggml_tensor* ru3_s2_alpha = nullptr;      // [512]
    ggml_tensor* ru3_c2_w = nullptr;          // [1, 512, 512]
    ggml_tensor* ru3_c2_b = nullptr;          // [512]

    // Block 1: snake1 + ConvT1d (s=5) + 3×ResUnit
    ggml_tensor* b1_s1_alpha = nullptr;        // [256]
    ggml_tensor* b1_convt_w = nullptr;          // [256, 2560] w_perm for ConvT1d(256→128, k=10, s=5)
    ggml_tensor* b1_convt_b = nullptr;          // [128]
    ggml_tensor* b1_ru1_s1_a = nullptr; ggml_tensor* b1_ru1_c1_w = nullptr; ggml_tensor* b1_ru1_c1_b = nullptr;
    ggml_tensor* b1_ru1_s2_a = nullptr; ggml_tensor* b1_ru1_c2_w = nullptr; ggml_tensor* b1_ru1_c2_b = nullptr;
    ggml_tensor* b1_ru2_s1_a = nullptr; ggml_tensor* b1_ru2_c1_w = nullptr; ggml_tensor* b1_ru2_c1_b = nullptr;
    ggml_tensor* b1_ru2_s2_a = nullptr; ggml_tensor* b1_ru2_c2_w = nullptr; ggml_tensor* b1_ru2_c2_b = nullptr;
    ggml_tensor* b1_ru3_s1_a = nullptr; ggml_tensor* b1_ru3_c1_w = nullptr; ggml_tensor* b1_ru3_c1_b = nullptr;
    ggml_tensor* b1_ru3_s2_a = nullptr; ggml_tensor* b1_ru3_c2_w = nullptr; ggml_tensor* b1_ru3_c2_b = nullptr;

    // Block 2: snake1 + ConvT1d(s=4) + 3×ResUnit
    ggml_tensor* b2_s1_alpha = nullptr; ggml_tensor* b2_convt_w = nullptr; ggml_tensor* b2_convt_b = nullptr;
    ggml_tensor* b2_ru1_s1_a = nullptr; ggml_tensor* b2_ru1_c1_w = nullptr; ggml_tensor* b2_ru1_c1_b = nullptr;
    ggml_tensor* b2_ru1_s2_a = nullptr; ggml_tensor* b2_ru1_c2_w = nullptr; ggml_tensor* b2_ru1_c2_b = nullptr;
    ggml_tensor* b2_ru2_s1_a = nullptr; ggml_tensor* b2_ru2_c1_w = nullptr; ggml_tensor* b2_ru2_c1_b = nullptr;
    ggml_tensor* b2_ru2_s2_a = nullptr; ggml_tensor* b2_ru2_c2_w = nullptr; ggml_tensor* b2_ru2_c2_b = nullptr;
    ggml_tensor* b2_ru3_s1_a = nullptr; ggml_tensor* b2_ru3_c1_w = nullptr; ggml_tensor* b2_ru3_c1_b = nullptr;
    ggml_tensor* b2_ru3_s2_a = nullptr; ggml_tensor* b2_ru3_c2_w = nullptr; ggml_tensor* b2_ru3_c2_b = nullptr;

    // Block 3: snake1 + ConvT1d(s=2) + 3×ResUnit
    ggml_tensor* b3_s1_alpha = nullptr; ggml_tensor* b3_convt_w = nullptr; ggml_tensor* b3_convt_b = nullptr;
    ggml_tensor* b3_ru1_s1_a = nullptr; ggml_tensor* b3_ru1_c1_w = nullptr; ggml_tensor* b3_ru1_c1_b = nullptr;
    ggml_tensor* b3_ru1_s2_a = nullptr; ggml_tensor* b3_ru1_c2_w = nullptr; ggml_tensor* b3_ru1_c2_b = nullptr;
    ggml_tensor* b3_ru2_s1_a = nullptr; ggml_tensor* b3_ru2_c1_w = nullptr; ggml_tensor* b3_ru2_c1_b = nullptr;
    ggml_tensor* b3_ru2_s2_a = nullptr; ggml_tensor* b3_ru2_c2_w = nullptr; ggml_tensor* b3_ru2_c2_b = nullptr;
    ggml_tensor* b3_ru3_s1_a = nullptr; ggml_tensor* b3_ru3_c1_w = nullptr; ggml_tensor* b3_ru3_c1_b = nullptr;
    ggml_tensor* b3_ru3_s2_a = nullptr; ggml_tensor* b3_ru3_c2_w = nullptr; ggml_tensor* b3_ru3_c2_b = nullptr;

    // Output layer: snake1 → conv2 → tanh
    // Block 4: snake1 + ConvT1d(s=3) + 3×ResUnit
    ggml_tensor* b4_s1_alpha = nullptr; ggml_tensor* b4_convt_w = nullptr; ggml_tensor* b4_convt_b = nullptr;
    ggml_tensor* b4_ru1_s1_a = nullptr; ggml_tensor* b4_ru1_c1_w = nullptr; ggml_tensor* b4_ru1_c1_b = nullptr;
    ggml_tensor* b4_ru1_s2_a = nullptr; ggml_tensor* b4_ru1_c2_w = nullptr; ggml_tensor* b4_ru1_c2_b = nullptr;
    ggml_tensor* b4_ru2_s1_a = nullptr; ggml_tensor* b4_ru2_c1_w = nullptr; ggml_tensor* b4_ru2_c1_b = nullptr;
    ggml_tensor* b4_ru2_s2_a = nullptr; ggml_tensor* b4_ru2_c2_w = nullptr; ggml_tensor* b4_ru2_c2_b = nullptr;
    ggml_tensor* b4_ru3_s1_a = nullptr; ggml_tensor* b4_ru3_c1_w = nullptr; ggml_tensor* b4_ru3_c1_b = nullptr;
    ggml_tensor* b4_ru3_s2_a = nullptr; ggml_tensor* b4_ru3_c2_w = nullptr; ggml_tensor* b4_ru3_c2_b = nullptr;

    // Output layer: snake1 → conv2 → tanh
    ggml_tensor* out_s1_alpha = nullptr;     // [32]
    ggml_tensor* out_conv2_w  = nullptr;     // [7, 32, 1]
    ggml_tensor* out_conv2_b  = nullptr;     // [1]

    // Text embedding
    ggml_tensor* token_embd = nullptr;        // [vocab_size, d_model]

    // Fused multi-codebook embedding
    ggml_tensor* fused_embed = nullptr;       // [8*1026, 2560]

    // Backbone Layer 0 (individual fields for debug convenience)
    ggml_tensor* l0_attn_norm   = nullptr;    // [2560]  RMSNorm
    ggml_tensor* l0_q_norm      = nullptr;    // [128]   QK-Norm
    ggml_tensor* l0_k_norm      = nullptr;    // [128]   QK-Norm
    ggml_tensor* l0_attn_q      = nullptr;    // [4096, 2560]
    ggml_tensor* l0_attn_k      = nullptr;    // [1024, 2560]
    ggml_tensor* l0_attn_v      = nullptr;    // [1024, 2560]
    ggml_tensor* l0_attn_o      = nullptr;    // [2560, 4096]
    ggml_tensor* l0_ffn_norm    = nullptr;    // [2560]  RMSNorm
    ggml_tensor* l0_ffn_gate    = nullptr;    // [9728, 2560]
    ggml_tensor* l0_ffn_up      = nullptr;    // [9728, 2560]
    ggml_tensor* l0_ffn_down    = nullptr;    // [2560, 9728]

    struct higgs_test_layer {
        ggml_tensor* attn_norm   = nullptr;   // RMSNorm [2560]
        ggml_tensor* q_norm      = nullptr;   // QK-Norm [128]
        ggml_tensor* k_norm      = nullptr;   // QK-Norm [128]
        ggml_tensor* attn_q      = nullptr;   // [4096, 2560]
        ggml_tensor* attn_k      = nullptr;   // [1024, 2560]
        ggml_tensor* attn_v      = nullptr;   // [1024, 2560]
        ggml_tensor* attn_o      = nullptr;   // [2560, 4096]
        ggml_tensor* ffn_norm    = nullptr;   // RMSNorm [2560]
        ggml_tensor* ffn_gate    = nullptr;   // [9728, 2560]
        ggml_tensor* ffn_up      = nullptr;   // [9728, 2560]
        ggml_tensor* ffn_down    = nullptr;   // [2560, 9728]
    };
    higgs_test_layer layer[36];

    // Backbone output
    ggml_tensor* output_norm = nullptr;       // [2560]  final RMSNorm
    ggml_tensor* fused_head  = nullptr;       // [8*1026, 2560]  output projection

    // Backbone hyperparams
    int n_heads   = 32;
    int n_kv_heads = 8;
    int head_dim  = 128;
    float rope_theta = 1000000.0f;

    // Special token IDs (hardcoded from notebook; GGUF metadata overrides if present)
    int tok_tts       = 151667;
    int tok_ref_text  = 151680;
    int tok_ref_audio = 151679;
    int tok_text      = 151672;
    int tok_audio     = 151670;

    // Vocab (BPE)
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
    std::map<std::string, ggml_tensor*> tensors;
};

bool higgs_test_load(const char* gguf_path, higgs_test_model* m);
bool higgs_test_load_vocab(const char* gguf_path, higgs_test_model* m);
void higgs_test_free(higgs_test_model* m);

#ifdef __cplusplus
}
#endif
