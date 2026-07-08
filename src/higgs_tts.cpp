// higgs_tts.cpp — Higgs TTS model loading

#include "higgs_tts.h"

#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// ── Load ───────────────────────────────────────────────────────────────────────

bool higgs_test_load(const char* gguf_path, higgs_test_model* m) {
    if (!m || !gguf_path) return false;

    // backend_cpu is always created first (qwen3_tts pattern)
    m->backend_cpu = ggml_backend_cpu_init();
    if (!m->backend_cpu) { std::fprintf(stderr, "higgs_test: cpu backend init fail\n"); return false; }

    m->backend = ggml_backend_init_best();
    if (!m->backend) m->backend = m->backend_cpu;

    const bool has_gpu = (m->backend != m->backend_cpu);
    ggml_backend_t weight_backend = m->backend;

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(gguf_path, weight_backend, "higgs", wl)) {
        std::fprintf(stderr, "higgs_test: weight load fail\n");
        return false;
    }
    m->buf = wl.buf;
    m->tensors = std::move(wl.tensors);

    auto& t = m->tensors;
    auto req = [&](const char* name) { return core_gguf::require(t, name, "higgs_test"); };

    for (int i = 0; i < m->N; i++) {
        auto s = std::string("codec.quant.") + std::to_string(i) + ".";
        m->quant[i].proj_in_w  = req((s + "project_in.weight").c_str());
        m->quant[i].proj_in_b  = req((s + "project_in.bias").c_str());
        m->quant[i].codebook    = req((s + "codebook.embed").c_str());
        m->quant[i].proj_out_w  = req((s + "project_out.weight").c_str());
        m->quant[i].proj_out_b  = req((s + "project_out.bias").c_str());
    }
    m->fc2_w  = req("codec.fc2.weight");
    m->fc2_b  = req("codec.fc2.bias");
    m->fc_w   = req("codec.fc.weight");
    m->fc_b   = req("codec.fc.bias");
    m->fc1_w  = req("codec.fc1.weight");
    m->fc1_b  = req("codec.fc1.bias");
    m->conv1_w = req("codec.ac_dec.conv1.weight");
    m->conv1_b = req("codec.ac_dec.conv1.bias");
    m->snake1_alpha = req("codec.ac_dec.block.0.snake1.alpha");
    m->convt1_w = req("codec.ac_dec.block.0.conv_t1.weight");
    m->convt1_b = req("codec.ac_dec.block.0.conv_t1.bias");
    m->ru1_s1_alpha = req("codec.ac_dec.block.0.res_unit1.snake1.alpha");
    m->ru1_c1_w     = req("codec.ac_dec.block.0.res_unit1.conv1.weight");
    m->ru1_c1_b     = req("codec.ac_dec.block.0.res_unit1.conv1.bias");
    m->ru1_s2_alpha = req("codec.ac_dec.block.0.res_unit1.snake2.alpha");
    m->ru1_c2_w     = req("codec.ac_dec.block.0.res_unit1.conv2.weight");
    m->ru1_c2_b     = req("codec.ac_dec.block.0.res_unit1.conv2.bias");
    m->ru2_s1_alpha = req("codec.ac_dec.block.0.res_unit2.snake1.alpha");
    m->ru2_c1_w     = req("codec.ac_dec.block.0.res_unit2.conv1.weight");
    m->ru2_c1_b     = req("codec.ac_dec.block.0.res_unit2.conv1.bias");
    m->ru2_s2_alpha = req("codec.ac_dec.block.0.res_unit2.snake2.alpha");
    m->ru2_c2_w     = req("codec.ac_dec.block.0.res_unit2.conv2.weight");
    m->ru2_c2_b     = req("codec.ac_dec.block.0.res_unit2.conv2.bias");
    m->ru3_s1_alpha = req("codec.ac_dec.block.0.res_unit3.snake1.alpha");
    m->ru3_c1_w     = req("codec.ac_dec.block.0.res_unit3.conv1.weight");
    m->ru3_c1_b     = req("codec.ac_dec.block.0.res_unit3.conv1.bias");
    m->ru3_s2_alpha = req("codec.ac_dec.block.0.res_unit3.snake2.alpha");
    m->ru3_c2_w     = req("codec.ac_dec.block.0.res_unit3.conv2.weight");
    m->ru3_c2_b     = req("codec.ac_dec.block.0.res_unit3.conv2.bias");

    // Block 1
    auto& m_ = *m;
    m_.b1_s1_alpha = req("codec.ac_dec.block.1.snake1.alpha");
    m_.b1_convt_w  = req("codec.ac_dec.block.1.conv_t1.weight");
    m_.b1_convt_b  = req("codec.ac_dec.block.1.conv_t1.bias");
    m_.b1_ru1_s1_a = req("codec.ac_dec.block.1.res_unit1.snake1.alpha");
    m_.b1_ru1_c1_w = req("codec.ac_dec.block.1.res_unit1.conv1.weight");
    m_.b1_ru1_c1_b = req("codec.ac_dec.block.1.res_unit1.conv1.bias");
    m_.b1_ru1_s2_a = req("codec.ac_dec.block.1.res_unit1.snake2.alpha");
    m_.b1_ru1_c2_w = req("codec.ac_dec.block.1.res_unit1.conv2.weight");
    m_.b1_ru1_c2_b = req("codec.ac_dec.block.1.res_unit1.conv2.bias");
    m_.b1_ru2_s1_a = req("codec.ac_dec.block.1.res_unit2.snake1.alpha");
    m_.b1_ru2_c1_w = req("codec.ac_dec.block.1.res_unit2.conv1.weight");
    m_.b1_ru2_c1_b = req("codec.ac_dec.block.1.res_unit2.conv1.bias");
    m_.b1_ru2_s2_a = req("codec.ac_dec.block.1.res_unit2.snake2.alpha");
    m_.b1_ru2_c2_w = req("codec.ac_dec.block.1.res_unit2.conv2.weight");
    m_.b1_ru2_c2_b = req("codec.ac_dec.block.1.res_unit2.conv2.bias");
    m_.b1_ru3_s1_a = req("codec.ac_dec.block.1.res_unit3.snake1.alpha");
    m_.b1_ru3_c1_w = req("codec.ac_dec.block.1.res_unit3.conv1.weight");
    m_.b1_ru3_c1_b = req("codec.ac_dec.block.1.res_unit3.conv1.bias");
    m_.b1_ru3_s2_a = req("codec.ac_dec.block.1.res_unit3.snake2.alpha");
    m_.b1_ru3_c2_w = req("codec.ac_dec.block.1.res_unit3.conv2.weight");
    m_.b1_ru3_c2_b = req("codec.ac_dec.block.1.res_unit3.conv2.bias");

    // Block 2
    m_.b2_s1_alpha  = req("codec.ac_dec.block.2.snake1.alpha");
    m_.b2_convt_w   = req("codec.ac_dec.block.2.conv_t1.weight");
    m_.b2_convt_b   = req("codec.ac_dec.block.2.conv_t1.bias");
    m_.b2_ru1_s1_a  = req("codec.ac_dec.block.2.res_unit1.snake1.alpha");
    m_.b2_ru1_c1_w  = req("codec.ac_dec.block.2.res_unit1.conv1.weight");
    m_.b2_ru1_c1_b  = req("codec.ac_dec.block.2.res_unit1.conv1.bias");
    m_.b2_ru1_s2_a  = req("codec.ac_dec.block.2.res_unit1.snake2.alpha");
    m_.b2_ru1_c2_w  = req("codec.ac_dec.block.2.res_unit1.conv2.weight");
    m_.b2_ru1_c2_b  = req("codec.ac_dec.block.2.res_unit1.conv2.bias");
    m_.b2_ru2_s1_a  = req("codec.ac_dec.block.2.res_unit2.snake1.alpha");
    m_.b2_ru2_c1_w  = req("codec.ac_dec.block.2.res_unit2.conv1.weight");
    m_.b2_ru2_c1_b  = req("codec.ac_dec.block.2.res_unit2.conv1.bias");
    m_.b2_ru2_s2_a  = req("codec.ac_dec.block.2.res_unit2.snake2.alpha");
    m_.b2_ru2_c2_w  = req("codec.ac_dec.block.2.res_unit2.conv2.weight");
    m_.b2_ru2_c2_b  = req("codec.ac_dec.block.2.res_unit2.conv2.bias");
    m_.b2_ru3_s1_a  = req("codec.ac_dec.block.2.res_unit3.snake1.alpha");
    m_.b2_ru3_c1_w  = req("codec.ac_dec.block.2.res_unit3.conv1.weight");
    m_.b2_ru3_c1_b  = req("codec.ac_dec.block.2.res_unit3.conv1.bias");
    m_.b2_ru3_s2_a  = req("codec.ac_dec.block.2.res_unit3.snake2.alpha");
    m_.b2_ru3_c2_w  = req("codec.ac_dec.block.2.res_unit3.conv2.weight");
    m_.b2_ru3_c2_b  = req("codec.ac_dec.block.2.res_unit3.conv2.bias");

    // Block 3
    m_.b3_s1_alpha  = req("codec.ac_dec.block.3.snake1.alpha");
    m_.b3_convt_w   = req("codec.ac_dec.block.3.conv_t1.weight");
    m_.b3_convt_b   = req("codec.ac_dec.block.3.conv_t1.bias");
    m_.b3_ru1_s1_a  = req("codec.ac_dec.block.3.res_unit1.snake1.alpha");
    m_.b3_ru1_c1_w  = req("codec.ac_dec.block.3.res_unit1.conv1.weight");
    m_.b3_ru1_c1_b  = req("codec.ac_dec.block.3.res_unit1.conv1.bias");
    m_.b3_ru1_s2_a  = req("codec.ac_dec.block.3.res_unit1.snake2.alpha");
    m_.b3_ru1_c2_w  = req("codec.ac_dec.block.3.res_unit1.conv2.weight");
    m_.b3_ru1_c2_b  = req("codec.ac_dec.block.3.res_unit1.conv2.bias");
    m_.b3_ru2_s1_a  = req("codec.ac_dec.block.3.res_unit2.snake1.alpha");
    m_.b3_ru2_c1_w  = req("codec.ac_dec.block.3.res_unit2.conv1.weight");
    m_.b3_ru2_c1_b  = req("codec.ac_dec.block.3.res_unit2.conv1.bias");
    m_.b3_ru2_s2_a  = req("codec.ac_dec.block.3.res_unit2.snake2.alpha");
    m_.b3_ru2_c2_w  = req("codec.ac_dec.block.3.res_unit2.conv2.weight");
    m_.b3_ru2_c2_b  = req("codec.ac_dec.block.3.res_unit2.conv2.bias");
    m_.b3_ru3_s1_a  = req("codec.ac_dec.block.3.res_unit3.snake1.alpha");
    m_.b3_ru3_c1_w  = req("codec.ac_dec.block.3.res_unit3.conv1.weight");
    m_.b3_ru3_c1_b  = req("codec.ac_dec.block.3.res_unit3.conv1.bias");
    m_.b3_ru3_s2_a  = req("codec.ac_dec.block.3.res_unit3.snake2.alpha");
    m_.b3_ru3_c2_w  = req("codec.ac_dec.block.3.res_unit3.conv2.weight");
    m_.b3_ru3_c2_b  = req("codec.ac_dec.block.3.res_unit3.conv2.bias");

    // Block 4
    m_.b4_s1_alpha  = req("codec.ac_dec.block.4.snake1.alpha");
    m_.b4_convt_w   = req("codec.ac_dec.block.4.conv_t1.weight");
    m_.b4_convt_b   = req("codec.ac_dec.block.4.conv_t1.bias");
    m_.b4_ru1_s1_a  = req("codec.ac_dec.block.4.res_unit1.snake1.alpha");
    m_.b4_ru1_c1_w  = req("codec.ac_dec.block.4.res_unit1.conv1.weight");
    m_.b4_ru1_c1_b  = req("codec.ac_dec.block.4.res_unit1.conv1.bias");
    m_.b4_ru1_s2_a  = req("codec.ac_dec.block.4.res_unit1.snake2.alpha");
    m_.b4_ru1_c2_w  = req("codec.ac_dec.block.4.res_unit1.conv2.weight");
    m_.b4_ru1_c2_b  = req("codec.ac_dec.block.4.res_unit1.conv2.bias");
    m_.b4_ru2_s1_a  = req("codec.ac_dec.block.4.res_unit2.snake1.alpha");
    m_.b4_ru2_c1_w  = req("codec.ac_dec.block.4.res_unit2.conv1.weight");
    m_.b4_ru2_c1_b  = req("codec.ac_dec.block.4.res_unit2.conv1.bias");
    m_.b4_ru2_s2_a  = req("codec.ac_dec.block.4.res_unit2.snake2.alpha");
    m_.b4_ru2_c2_w  = req("codec.ac_dec.block.4.res_unit2.conv2.weight");
    m_.b4_ru2_c2_b  = req("codec.ac_dec.block.4.res_unit2.conv2.bias");
    m_.b4_ru3_s1_a  = req("codec.ac_dec.block.4.res_unit3.snake1.alpha");
    m_.b4_ru3_c1_w  = req("codec.ac_dec.block.4.res_unit3.conv1.weight");
    m_.b4_ru3_c1_b  = req("codec.ac_dec.block.4.res_unit3.conv1.bias");
    m_.b4_ru3_s2_a  = req("codec.ac_dec.block.4.res_unit3.snake2.alpha");
    m_.b4_ru3_c2_w  = req("codec.ac_dec.block.4.res_unit3.conv2.weight");
    m_.b4_ru3_c2_b  = req("codec.ac_dec.block.4.res_unit3.conv2.bias");

    // Output layer
    m_.out_s1_alpha = req("codec.ac_dec.snake1.alpha");
    m_.out_conv2_w  = req("codec.ac_dec.conv2.weight");
    m_.out_conv2_b  = req("codec.ac_dec.conv2.bias");

    // ── Codec Encoder ─────────────────────────────────────────────────────
    // Acoustic encoder (DAC): conv1 + 5 blocks + conv2 + snake
    m_.ac_enc_conv1_w = req("codec.ac_enc.conv1.weight");
    m_.ac_enc_conv1_b = req("codec.ac_enc.conv1.bias");
    for (int i = 0; i < 5; i++) {
        char k[64];
        snprintf(k, sizeof(k), "codec.ac_enc.block.%d.conv1.weight", i);
        m_.ac_enc_blocks[i].conv1_w = req(k);
        snprintf(k, sizeof(k), "codec.ac_enc.block.%d.conv1.bias", i);
        m_.ac_enc_blocks[i].conv1_b = req(k);
        snprintf(k, sizeof(k), "codec.ac_enc.block.%d.snake1.alpha", i);
        m_.ac_enc_blocks[i].snake1_alpha = req(k);
        for (int j = 0; j < 3; j++) {
            snprintf(k, sizeof(k), "codec.ac_enc.block.%d.res_unit%d.conv1.weight", i, j+1);
            m_.ac_enc_blocks[i].ru[j].conv1_w = req(k);
            snprintf(k, sizeof(k), "codec.ac_enc.block.%d.res_unit%d.conv1.bias", i, j+1);
            m_.ac_enc_blocks[i].ru[j].conv1_b = req(k);
            snprintf(k, sizeof(k), "codec.ac_enc.block.%d.res_unit%d.conv2.weight", i, j+1);
            m_.ac_enc_blocks[i].ru[j].conv2_w = req(k);
            snprintf(k, sizeof(k), "codec.ac_enc.block.%d.res_unit%d.conv2.bias", i, j+1);
            m_.ac_enc_blocks[i].ru[j].conv2_b = req(k);
            snprintf(k, sizeof(k), "codec.ac_enc.block.%d.res_unit%d.snake1.alpha", i, j+1);
            m_.ac_enc_blocks[i].ru[j].snake1_alpha = req(k);
            snprintf(k, sizeof(k), "codec.ac_enc.block.%d.res_unit%d.snake2.alpha", i, j+1);
            m_.ac_enc_blocks[i].ru[j].snake2_alpha = req(k);
        }
    }
    m_.ac_enc_conv2_w = req("codec.ac_enc.conv2.weight");
    m_.ac_enc_conv2_b = req("codec.ac_enc.conv2.bias");
    m_.ac_enc_snake1 = req("codec.ac_enc.snake1.alpha");

    // Semantic encoder: conv + 2 blocks (strides=[1,1], dilations=[1,1])
    m_.enc_sem_conv_w = req("codec.enc_sem.conv.weight");

    // // Dump ResUnit[0] conv1_w for verification
    // {
    //     auto dump_w = [&](ggml_tensor* t, const char* name) {
    //         int n = (int)ggml_nelements(t);
    //         std::vector<uint8_t> raw(ggml_nbytes(t));
    //         ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    //         std::vector<float> fbuf(n);
    //         if (t->type == GGML_TYPE_F16)
    //             ggml_fp16_to_fp32_row((ggml_fp16_t*)raw.data(), fbuf.data(), n);
    //         else memcpy(fbuf.data(), raw.data(), n * sizeof(float));
    //         std::printf("%s ne=[%lld,%lld,%lld] type=%s first10:",
    //             name, t->ne[0], t->ne[1], t->ne[2], ggml_type_name(t->type));
    //         for (int i = 0; i < std::min(10, n); i++) std::printf(" %.6f", fbuf[i]);
    //         std::printf("\n");
    //     };
    //     dump_w(m_.enc_sem_blocks[0].ru[0].conv1_w, "ru0_conv1_w");
    //     dump_w(m_.enc_sem_blocks[0].ru[0].conv2_w, "ru0_conv2_w");
    // }

    for (int i = 0; i < 2; i++) {
        char k[64];
        snprintf(k, sizeof(k), "codec.enc_sem.blk.%d.conv.weight", i);
        m_.enc_sem_blocks[i].conv_w = req(k);
        snprintf(k, sizeof(k), "codec.enc_sem.blk.%d.conv.bias", i);
        m_.enc_sem_blocks[i].conv_b = req(k);
        m_.enc_sem_blocks[i].ru.resize(2);
        for (int j = 0; j < 2; j++) {
            snprintf(k, sizeof(k), "codec.enc_sem.blk.%d.ru.%d.conv1.weight", i, j);
            m_.enc_sem_blocks[i].ru[j].conv1_w = req(k);
            // conv1: bias=False, no bias tensor
            snprintf(k, sizeof(k), "codec.enc_sem.blk.%d.ru.%d.conv2.weight", i, j);
            m_.enc_sem_blocks[i].ru[j].conv2_w = req(k);
            // conv2: bias=False, 1x1 conv
        }
    }

    // Hubert semantic model
    // Feature extractor: 7 Conv1d → GroupNorm → GELU layers
    m_.sem.fe.resize(7);
    for (int i = 0; i < 7; i++) {
        char k[64];
        snprintf(k, sizeof(k), "codec.sem.fe.cv.%d.conv.weight", i);
        m_.sem.fe[i].conv_w = req(k);
        // conv.bias: bias=False, not in GGUF
        // layer_norm → ln by convert script; only layer 0 has GroupNorm
        snprintf(k, sizeof(k), "codec.sem.fe.cv.%d.ln.weight", i);
        m_.sem.fe[i].norm_w = core_gguf::try_get(t, k);
        snprintf(k, sizeof(k), "codec.sem.fe.cv.%d.ln.bias", i);
        m_.sem.fe[i].norm_b = core_gguf::try_get(t, k);
    }
    // Feature projection
    m_.sem.fp_w    = req("codec.sem.fp.projection.weight");
    m_.sem.fp_b    = req("codec.sem.fp.projection.bias");
    m_.sem.fp_ln_w = req("codec.sem.fp.ln.weight");
    m_.sem.fp_ln_b = req("codec.sem.fp.ln.bias");

    // Positional conv embedding (grouped Conv1d with weight_norm)
    m_.sem.pce_orig0 = req("codec.sem.encoder.pce.conv.pm.weight.orig0");
    m_.sem.pce_orig1 = req("codec.sem.encoder.pce.conv.pm.weight.orig1");
    m_.sem.pce_bias  = core_gguf::try_get(t, "codec.sem.encoder.pce.conv.bias");
    if (m_.sem.pce_bias) {
        int nb = (int)ggml_nelements(m_.sem.pce_bias);
        std::vector<float> bias_dump(nb);
        ggml_backend_tensor_get(m_.sem.pce_bias, bias_dump.data(), 0, nb * sizeof(float));
//         // std::printf("pce_bias ne=[%lld] first10:", m_.sem.pce_bias->ne[0]);
//         for (int i = 0; i < 10; i++) std::printf(" %.6f", bias_dump[i]);
//         // std::printf("\n");
    }
    // Pre-fuse weight_norm on CPU: w = orig0 * orig1 / ||orig1||
    {
        int K = (int)m_.sem.pce_orig0->ne[0];   // 128
        int Cg = (int)m_.sem.pce_orig1->ne[1];  // 48
        int OC = (int)m_.sem.pce_orig1->ne[2];  // 768

        // Read orig0 [K, 1, 1] → g[K] (may be F16)
        std::vector<float> g(K);
        {
            std::vector<uint8_t> raw(ggml_nbytes(m_.sem.pce_orig0));
            ggml_backend_tensor_get(m_.sem.pce_orig0, raw.data(), 0, raw.size());
            if (m_.sem.pce_orig0->type == GGML_TYPE_F16)
                ggml_fp16_to_fp32_row((ggml_fp16_t*)raw.data(), g.data(), K);
            else
                memcpy(g.data(), raw.data(), K * sizeof(float));
        }
        // Read orig1 [K, Cg, OC] → v (may be F16)
        int n_v = K * Cg * OC;
        std::vector<float> v(n_v);
        {
            std::vector<uint8_t> raw(ggml_nbytes(m_.sem.pce_orig1));
            ggml_backend_tensor_get(m_.sem.pce_orig1, raw.data(), 0, raw.size());
            if (m_.sem.pce_orig1->type == GGML_TYPE_F16)
                ggml_fp16_to_fp32_row((ggml_fp16_t*)raw.data(), v.data(), n_v);
            else
                memcpy(v.data(), raw.data(), n_v * sizeof(float));
        }

        int n_w = K * Cg * OC;
        m_.sem.pce_weight_data.resize(n_w);
        // weight_norm: norm is over all (in_ch,out_ch) for each kernel position
        for (int i = 0; i < K; i++) {
            float norm_sq = 0.0f;
            for (int j = 0; j < Cg; j++)
                for (int k = 0; k < OC; k++)
                    norm_sq += v[i + j*K + k*K*Cg] * v[i + j*K + k*K*Cg];
            float norm = (norm_sq > 0.0f) ? sqrtf(norm_sq) : 1.0f;
            for (int j = 0; j < Cg; j++)
                for (int k = 0; k < OC; k++)
                    m_.sem.pce_weight_data[i + j*K + k*K*Cg] = g[i] * v[i + j*K + k*K*Cg] / norm;
        }
        // Dump fused weight shape + first 20
//         // std::printf("pce fused_w [%d,%d] (K*Cg=%d, OC=%d) first20:\n", K * Cg, OC, K * Cg, OC);
//         for (int i = 0; i < 20; i++) std::printf(" %.6f", m_.sem.pce_weight_data[i]);
//         // std::printf("\n");
    }

    // Encoder transformer layers (auto-detect count)
    for (int i = 0; i < 24; i++) {
        char k[128];
        snprintf(k, sizeof(k), "codec.sem.enc.%d.attn_q.weight", i);
        auto* probe = core_gguf::try_get(t, k);
        if (!probe) break;
        higgs_sem_layer L;
        L.attn_q_w = probe;
        snprintf(k, sizeof(k), "codec.sem.enc.%d.attn_q.bias", i);       L.attn_q_b   = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.attn_k.weight", i);     L.attn_k_w   = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.attn_k.bias", i);       L.attn_k_b   = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.attn_v.weight", i);     L.attn_v_w   = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.attn_v.bias", i);       L.attn_v_b   = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.attn_out.weight", i);   L.attn_out_w = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.attn_out.bias", i);     L.attn_out_b = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.ffn1.weight", i);       L.ffn1_w     = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.ffn1.bias", i);         L.ffn1_b     = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.ffn2.weight", i);       L.ffn2_w     = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.ffn2.bias", i);         L.ffn2_b     = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.ln.weight", i);         L.ln_w       = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.ln.bias", i);           L.ln_b       = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.fin_ln.weight", i);     L.fin_ln_w   = req(k);
        snprintf(k, sizeof(k), "codec.sem.enc.%d.fin_ln.bias", i);       L.fin_ln_b   = req(k);
        m_.sem.layers.push_back(L);
    }
    // Post-encoder layernorms
    m_.sem.post_ln_w = core_gguf::try_get(t, "codec.sem.encoder.ln.weight");
    m_.sem.post_ln_b = core_gguf::try_get(t, "codec.sem.encoder.ln.bias");
    m_.sem.top_ln_w  = core_gguf::try_get(t, "codec.sem.ln.weight");
    m_.sem.top_ln_b  = core_gguf::try_get(t, "codec.sem.ln.bias");

//     // std::printf("higgs_test: sem enc layers=%zu\n", m_.sem.layers.size());

    // Text embedding
    m_.token_embd = req("token_embd.weight");
    m_.fused_embed = req("fused_embed.weight");

    // Backbone Layer 0 (individual fields for debug)
    m_.l0_attn_norm = req("blk.0.attn_norm.weight");
    m_.l0_q_norm    = req("blk.0.attn_q_norm.weight");
    m_.l0_k_norm    = req("blk.0.attn_k_norm.weight");
    m_.l0_attn_q    = req("blk.0.attn_q.weight");
    m_.l0_attn_k    = req("blk.0.attn_k.weight");
    m_.l0_attn_v    = req("blk.0.attn_v.weight");
    m_.l0_attn_o    = req("blk.0.attn_output.weight");
    m_.l0_ffn_norm  = req("blk.0.ffn_norm.weight");
    m_.l0_ffn_gate  = req("blk.0.ffn_gate.weight");
    m_.l0_ffn_up    = req("blk.0.ffn_up.weight");
    m_.l0_ffn_down  = req("blk.0.ffn_down.weight");

    // Layers 0-35
    for (int i = 0; i < 36; i++) {
        auto& L = m_.layer[i];
        char k[64];
        snprintf(k, sizeof(k), "blk.%d.attn_norm.weight", i);
        L.attn_norm = req(k);
        snprintf(k, sizeof(k), "blk.%d.attn_q_norm.weight", i);
        L.q_norm = req(k);
        snprintf(k, sizeof(k), "blk.%d.attn_k_norm.weight", i);
        L.k_norm = req(k);
        snprintf(k, sizeof(k), "blk.%d.attn_q.weight", i);
        L.attn_q = req(k);
        snprintf(k, sizeof(k), "blk.%d.attn_k.weight", i);
        L.attn_k = req(k);
        snprintf(k, sizeof(k), "blk.%d.attn_v.weight", i);
        L.attn_v = req(k);
        snprintf(k, sizeof(k), "blk.%d.attn_output.weight", i);
        L.attn_o = req(k);
        snprintf(k, sizeof(k), "blk.%d.ffn_norm.weight", i);
        L.ffn_norm = req(k);
        snprintf(k, sizeof(k), "blk.%d.ffn_gate.weight", i);
        L.ffn_gate = req(k);
        snprintf(k, sizeof(k), "blk.%d.ffn_up.weight", i);
        L.ffn_up = req(k);
        snprintf(k, sizeof(k), "blk.%d.ffn_down.weight", i);
        L.ffn_down = req(k);
    }
    m_.output_norm = req("output_norm.weight");
    m_.fused_head  = req("fused_head.weight");

    // Scheduler: GPU+CPU when GPU available, CPU-only as fallback
    if (has_gpu) {
        ggml_backend_t backs[2] = { m->backend, m->backend_cpu };
        m->sched = ggml_backend_sched_new(backs, nullptr, 2, 16384, false, false);
    } else {
        ggml_backend_t backs[1] = { m->backend_cpu };
        m->sched = ggml_backend_sched_new(backs, nullptr, 1, 16384, false, false);
    }
    if (!m->sched) { std::fprintf(stderr, "higgs_test: sched fail\n"); return false; }
    m->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

//     // std::printf("higgs_test: loaded %zu tensors on %s, sched=%s\n", t.size(), ...);
    return true;
}

void higgs_test_free(higgs_test_model* m) {
    if (!m) return;
    if (m->sched) { ggml_backend_sched_free(m->sched); m->sched = nullptr; }
    if (m->buf) { ggml_backend_buffer_free(m->buf); m->buf = nullptr; }
    if (m->backend && m->backend != m->backend_cpu) {
        ggml_backend_free(m->backend);
    }
    m->backend = nullptr;
    if (m->backend_cpu) { ggml_backend_free(m->backend_cpu); m->backend_cpu = nullptr; }
}

bool higgs_test_load_vocab(const char* gguf_path, higgs_test_model* m) {
    gguf_context* meta = core_gguf::open_metadata(gguf_path);
    if (!meta) return false;

    auto toks = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    auto merges = core_gguf::kv_str_array(meta, "tokenizer.ggml.merges");

    // Override special token IDs from GGUF metadata if present (stored as higgs.token.*)
    auto stok = [&](const char* key, int& dst) {
        uint32_t v = core_gguf::kv_u32(meta, key, UINT32_MAX);
        if (v != UINT32_MAX) dst = (int)v;
    };
    stok("higgs.token.tts",       m->tok_tts);
    stok("higgs.token.ref_text",  m->tok_ref_text);
    stok("higgs.token.ref_audio", m->tok_ref_audio);
    stok("higgs.token.text",      m->tok_text);
    stok("higgs.token.audio",     m->tok_audio);

    core_gguf::free_metadata(meta);

    m->id_to_token = toks;
    for (int i = 0; i < (int)toks.size(); i++)
        m->token_to_id[toks[i]] = i;
    for (int i = 0; i < (int)merges.size(); i++)
        m->merge_rank[merges[i]] = i;

    return true;
}
