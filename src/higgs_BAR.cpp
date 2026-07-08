// higgs_BAR.cpp — Backbone AR: prefill embeddings + 36-layer backbone + AR decode loop

#include "higgs_BAR.h"
#include "higgs_tts.h"

#include "core/attention.h"
#include "core/ffn.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int AUDIO_PLACEHOLDER_ID = -100;
static const int BOC_ID = 1024;
static const int EOC_ID = 1025;

// ── apply_delay_pattern ───────────────────────────────────────────────────────
std::vector<int32_t> higgs_apply_delay_pattern(const int32_t* codes, int T, int N) {
    int L = T + N - 1;
    std::vector<int32_t> out(L * N, EOC_ID);
    for (int c = 0; c < N; c++) {
        for (int t = 0; t < c; t++)
            out[t * N + c] = BOC_ID;
        for (int t = 0; t < T; t++)
            out[(c + t) * N + c] = codes[t * N + c];
    }
    return out;
}

// ── build_prompt ──────────────────────────────────────────────────────────────
std::vector<int32_t> higgs_build_prompt(
    const higgs_test_model* m,
    const std::vector<int32_t>& target_tokens,
    const std::vector<int32_t>& ref_text_tokens,
    int L_audio, bool has_ref_audio)
{
    std::vector<int32_t> prompt;
    prompt.push_back(m->tok_tts);
    if (has_ref_audio) {
        if (!ref_text_tokens.empty() && m->tok_ref_text >= 0) {
            prompt.push_back(m->tok_ref_text);
            for (int t : ref_text_tokens) prompt.push_back(t);
        }
        if (m->tok_ref_audio >= 0) prompt.push_back(m->tok_ref_audio);
        for (int i = 0; i < L_audio; i++) prompt.push_back(AUDIO_PLACEHOLDER_ID);
    }
    prompt.push_back(m->tok_text);
    for (int t : target_tokens) prompt.push_back(t);
    prompt.push_back(m->tok_audio);
    return prompt;
}

// ── build_prefill_embeds ──────────────────────────────────────────────────────
static ggml_tensor* build_prefill_embeds(
    ggml_context* ctx,
    const int32_t* prompt_ids, int L_prompt,
    const int32_t* delayed_codes, int L_audio, int N,
    higgs_test_model* m,
    std::vector<ggml_tensor*>& inp)
{
    int D = (int)m->token_embd->ne[0];

    // Find audio block (contiguous -100)
    int pre_len = 0;
    while (pre_len < L_prompt && prompt_ids[pre_len] != AUDIO_PLACEHOLDER_ID) pre_len++;

    // Safe IDs (-100 → 0)
    std::vector<int32_t> safe_ids(L_prompt);
    for (int i = 0; i < L_prompt; i++) safe_ids[i] = (prompt_ids[i] == AUDIO_PLACEHOLDER_ID) ? 0 : prompt_ids[i];
    ggml_tensor* safe_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L_prompt);
    ggml_set_input(safe_t);
    inp.push_back(safe_t);
    ggml_tensor* text_emb = ggml_get_rows(ctx, m->token_embd, safe_t); // [D, L]

    // Fused audio embeddings
    ggml_tensor* audio_emb = nullptr;
    for (int c = 0; c < N; c++) {
        std::vector<int32_t> cb_idx(L_audio);
        for (int t = 0; t < L_audio; t++)
            cb_idx[t] = c * 1026 + delayed_codes[t * N + c];
        ggml_tensor* idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L_audio);
        ggml_set_input(idx_t);
        inp.push_back(idx_t);
        ggml_tensor* cb_emb = ggml_get_rows(ctx, m->fused_embed, idx_t);
        audio_emb = audio_emb ? ggml_add(ctx, audio_emb, cb_emb) : cb_emb;
    }

    // Mask: 0 at audio positions
    std::vector<float> mask(L_prompt);
    for (int i = 0; i < L_prompt; i++)
        mask[i] = (prompt_ids[i] == AUDIO_PLACEHOLDER_ID) ? 0.0f : 1.0f;
    ggml_tensor* mask_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, L_prompt);
    ggml_set_input(mask_t);
    inp.push_back(mask_t);
    ggml_tensor* masked_emb = ggml_mul(ctx, text_emb, ggml_reshape_2d(ctx, mask_t, 1, L_prompt));

    // Scatter audio_emb into audio positions
    ggml_tensor* audio_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L_prompt);
    ggml_tensor* zero = ggml_scale(ctx, audio_pad, 0.0f);
    size_t row_bytes = D * sizeof(float);
    ggml_tensor* audio_scattered = ggml_acc(ctx, zero, audio_emb,
        (int)zero->nb[1], (int)zero->nb[2], (int)zero->nb[3], pre_len * row_bytes);

    ggml_tensor* out = ggml_add(ctx, masked_emb, audio_scattered);
    ggml_set_name(out, "inputs_embeds");
    return out;
}

// ── helpers ───────────────────────────────────────────────────────────────────
static void sample_codes(const float* logits, int N, int Vcb, float temperature,
                          std::vector<int32_t>& codes_n) {
    codes_n.resize(N);
    for (int c = 0; c < N; c++) {
        const float* cb = logits + c * Vcb;
        std::vector<std::pair<float, int>> sorted(Vcb);
        for (int k = 0; k < Vcb; k++) sorted[k] = {cb[k] / temperature, k};
        int topk = std::min(50, Vcb);
        std::partial_sort(sorted.begin(), sorted.begin() + topk, sorted.end(),
                          std::greater<std::pair<float, int>>());
        float max_val = sorted[0].first;
        float sum_exp = 0.0f;
        for (int k = 0; k < topk; k++) {
            sorted[k].first = expf(sorted[k].first - max_val);
            sum_exp += sorted[k].first;
        }
        float r = (float)rand() / (float)RAND_MAX;
        float cum = 0.0f;
        int chosen = 0;
        for (int k = 0; k < topk; k++) {
            cum += sorted[k].first / sum_exp;
            if (r <= cum) { chosen = k; break; }
        }
        codes_n[c] = sorted[chosen].second;
    }
}

// ── trim trailing silence ───────────────────────────────────────────────────
void higgs_trim_trailing_silence(std::vector<float>& pcm, float db_threshold) {
    float amp_threshold = powf(10.0f, db_threshold / 20.0f);
    int n = (int)pcm.size();
    int cut = n;
    while (cut > 0 && fabsf(pcm[cut - 1]) < amp_threshold)
        cut--;
    if (cut < n) {
        // Leave a tiny fade: keep ~20ms of the trimmed silence as a soft tail
        int keep = std::min(n - cut, 480);  // 480 samples ≈ 20ms @ 24kHz
        cut += keep;
        if (cut > n) cut = n;
        pcm.resize(cut);
    }
}

// ── higgs_backbone_ar ─────────────────────────────────────────────────────────
bool higgs_backbone_ar(higgs_test_model* m, const int32_t* codes, int T_frames,
                        const int32_t* prompt_ids, int L_prompt,
                        float temperature, int seed,
                        std::vector<int32_t>& raw_codes, int& T_raw) {
    if (!m || !codes || !prompt_ids || T_frames < 1 || L_prompt < 1) return false;

    const int N = 8, Vcb = 1026, D = 2560;
    const int hd = 128, nh = 32, nkv_h = 8;
    const float eps = 1e-6f;
    srand(seed);

    // Apply delay pattern
    auto delayed = higgs_apply_delay_pattern(codes, T_frames, N);
    int L_audio = T_frames + N - 1;
    int L = L_prompt;

    // ── Allocate KV cache ───────────────────────────────────────────────────
    int max_ctx = 2048;
    ggml_init_params kv_ip = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context* kv_ctx = ggml_init(kv_ip);
    ggml_tensor* kv_k = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, hd, max_ctx, nkv_h, 36);
    ggml_tensor* kv_v = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, hd, max_ctx, nkv_h, 36);
    size_t kv_kb = ggml_nbytes(kv_k), kv_vb = ggml_nbytes(kv_v);
    ggml_backend_buffer_t kv_buf = ggml_backend_alloc_buffer(m->backend, kv_kb + kv_vb);
    char* kv_base = (char*)ggml_backend_buffer_get_base(kv_buf);
    ggml_backend_tensor_alloc(kv_buf, kv_k, kv_base);
    ggml_backend_tensor_alloc(kv_buf, kv_v, kv_base + kv_kb);

    // ── Prefill graph ───────────────────────────────────────────────────────
    ggml_init_params ip = { m->compute_meta.size(), m->compute_meta.data(), true };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) { ggml_backend_buffer_free(kv_buf); ggml_free(kv_ctx); return false; }

    std::vector<ggml_tensor*> inp;
    ggml_tensor* x = build_prefill_embeds(ctx, prompt_ids, L_prompt, delayed.data(), L_audio, N, m, inp);

    // Inputs
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_set_input(pos); inp.push_back(pos);
    ggml_tensor* prefill_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, L);
    ggml_set_input(prefill_mask); inp.push_back(prefill_mask);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
    const core_attn::KvSelfAttnParams kvp = {
        nh, nkv_h, hd, nh / nkv_h, 0, m->rope_theta, 32.0f, 1.0f,
        1.0f / sqrtf((float)hd), eps, core_attn::GQA_NATIVE,
    };

    for (int i = 0; i < 36; i++) {
        auto& ly = m->layer[i];
        ggml_tensor* residual = x;
        x = ggml_rms_norm(ctx, x, eps);
        x = ggml_mul(ctx, x, ggml_reshape_2d(ctx, ly.attn_norm, D, 1));
        x = core_attn::kv_self_attn(ctx, gf, x, ly.attn_q, ly.attn_k, ly.attn_v, ly.attn_o,
                                    ly.q_norm, ly.k_norm, pos, prefill_mask, kv_k, kv_v, i, 0, kvp);
        x = ggml_add(ctx, residual, x);
        residual = x;
        x = ggml_rms_norm(ctx, x, eps);
        x = ggml_mul(ctx, x, ggml_reshape_2d(ctx, ly.ffn_norm, D, 1));
        x = core_ffn::swiglu(ctx, x, ly.ffn_gate, ly.ffn_up, ly.ffn_down);
        x = ggml_add(ctx, residual, x);
    }
    x = ggml_rms_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, ggml_reshape_2d(ctx, m->output_norm, D, 1));
    ggml_tensor* last_hidden = ggml_view_1d(ctx, x, D, (L - 1) * x->nb[1]);
    ggml_tensor* hidden_2d = ggml_reshape_2d(ctx, last_hidden, D, 1);
    ggml_tensor* logits = ggml_mul_mat(ctx, m->fused_head, hidden_2d);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    // Alloc & upload inputs
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "BAR: prefill alloc fail\n");
        ggml_free(ctx); ggml_backend_buffer_free(kv_buf); ggml_free(kv_ctx); return false;
    }
    {
        int k = 0;
        // safe_ids
        std::vector<int32_t> safe_ids(L_prompt);
        for (int i = 0; i < L_prompt; i++) safe_ids[i] = (prompt_ids[i] == AUDIO_PLACEHOLDER_ID) ? 0 : prompt_ids[i];
        ggml_backend_tensor_set(inp[k], safe_ids.data(), 0, L_prompt * sizeof(int32_t)); k++;
        // codebook indices ×8
        for (int c = 0; c < N; c++) {
            std::vector<int32_t> cb_idx(L_audio);
            for (int t = 0; t < L_audio; t++) cb_idx[t] = c * Vcb + delayed[t * N + c];
            ggml_backend_tensor_set(inp[k], cb_idx.data(), 0, L_audio * sizeof(int32_t)); k++;
        }
        // float mask
        std::vector<float> mask(L_prompt);
        for (int i = 0; i < L_prompt; i++) mask[i] = (prompt_ids[i] == AUDIO_PLACEHOLDER_ID) ? 0.0f : 1.0f;
        ggml_backend_tensor_set(inp[k], mask.data(), 0, L_prompt * sizeof(float)); k++;
        // positions
        std::vector<int32_t> pos_data(L);
        for (int i = 0; i < L; i++) pos_data[i] = i;
        ggml_backend_tensor_set(inp[k], pos_data.data(), 0, L * sizeof(int32_t)); k++;
        // causal mask
        std::vector<ggml_fp16_t> mask_data(L * L);
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                mask_data[i * L + j] = (j <= i) ? 0 : ggml_fp32_to_fp16(-INFINITY);
        ggml_backend_tensor_set(inp[k], mask_data.data(), 0, L * L * sizeof(ggml_fp16_t)); k++;
    }

    if (ggml_backend_sched_graph_compute(m->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "BAR: prefill compute fail\n");
        ggml_free(ctx); ggml_backend_buffer_free(kv_buf); ggml_free(kv_ctx); return false;
    }

    // Read prefill outputs
    std::vector<float> logits_buf(N * Vcb);
    ggml_backend_tensor_get(logits, logits_buf.data(), 0, N * Vcb * sizeof(float));
    std::vector<float> cur_hidden(D);
    ggml_backend_tensor_get(last_hidden, cur_hidden.data(), 0, D * sizeof(float));
    ggml_free(ctx);

    // ── AR Decode Loop ──────────────────────────────────────────────────────
    int n_past = L;
    int max_steps = 300;
    int delay_count = 0;
    int eoc_countdown = -1;
    std::vector<std::vector<int32_t>> all_codes;

    for (int step = 0; step < max_steps; step++) {
        // Sample
        std::vector<int32_t> codes_n;
        sample_codes(logits_buf.data(), N, Vcb, temperature, codes_n);

        // Delay pattern + EOC
        if (delay_count < N) {
            int next_cb = delay_count + 1;
            if (next_cb < N)
                for (int c = next_cb; c < N; c++) codes_n[c] = BOC_ID;
            delay_count++;
        } else if (eoc_countdown >= 0) {
            eoc_countdown--;
        } else if (codes_n[0] == EOC_ID) {
            if (N <= 2) break;
            eoc_countdown = N - 2;
        }
        all_codes.push_back(codes_n);
        if (eoc_countdown == 0) break;

        // Build step graph
        ggml_init_params step_ip = { m->compute_meta.size(), m->compute_meta.data(), true };
        ggml_context* step_ctx = ggml_init(step_ip);
        ggml_cgraph* step_gf = ggml_new_graph_custom(step_ctx, 2048, false);

        ggml_tensor* cb_ids[8];
        for (int c = 0; c < N; c++) {
            cb_ids[c] = ggml_new_tensor_1d(step_ctx, GGML_TYPE_I32, 1);
            ggml_set_input(cb_ids[c]);
        }
        ggml_tensor* step_emb = nullptr;
        for (int c = 0; c < N; c++) {
            ggml_tensor* cb_emb = ggml_get_rows(step_ctx, m->fused_embed, cb_ids[c]);
            step_emb = step_emb ? ggml_add(step_ctx, step_emb, cb_emb) : cb_emb;
        }
        ggml_tensor* pos_t = ggml_new_tensor_1d(step_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(pos_t);

        const core_attn::KvSelfAttnParams kvp_step = {
            nh, nkv_h, hd, nh / nkv_h, 0, m->rope_theta, 32.0f, 1.0f,
            1.0f / sqrtf((float)hd), eps, core_attn::GQA_NATIVE,
        };

        ggml_tensor* cur = step_emb;
        for (int i = 0; i < 36; i++) {
            auto& ly = m->layer[i];
            ggml_tensor* residual = cur;
            cur = ggml_rms_norm(step_ctx, cur, eps);
            cur = ggml_mul(step_ctx, cur, ggml_reshape_2d(step_ctx, ly.attn_norm, D, 1));
            cur = core_attn::kv_self_attn(step_ctx, step_gf, cur, ly.attn_q, ly.attn_k, ly.attn_v, ly.attn_o,
                                          ly.q_norm, ly.k_norm, pos_t, nullptr, kv_k, kv_v, i, n_past, kvp_step);
            cur = ggml_add(step_ctx, residual, cur);
            residual = cur;
            cur = ggml_rms_norm(step_ctx, cur, eps);
            cur = ggml_mul(step_ctx, cur, ggml_reshape_2d(step_ctx, ly.ffn_norm, D, 1));
            cur = core_ffn::swiglu(step_ctx, cur, ly.ffn_gate, ly.ffn_up, ly.ffn_down);
            cur = ggml_add(step_ctx, residual, cur);
        }
        cur = ggml_rms_norm(step_ctx, cur, eps);
        cur = ggml_mul(step_ctx, cur, ggml_reshape_2d(step_ctx, m->output_norm, D, 1));
        ggml_tensor* step_logits = ggml_mul_mat(step_ctx, m->fused_head, cur);
        ggml_set_output(step_logits);
        ggml_build_forward_expand(step_gf, step_logits);

        ggml_backend_sched_reset(m->sched);
        if (!ggml_backend_sched_alloc_graph(m->sched, step_gf)) {
            fprintf(stderr, "BAR: step %d alloc fail\n", step); break;
        }
        for (int c = 0; c < N; c++) {
            int idx = c * Vcb + codes_n[c];
            ggml_backend_tensor_set(cb_ids[c], &idx, 0, sizeof(int32_t));
        }
        ggml_backend_tensor_set(pos_t, &n_past, 0, sizeof(int32_t));
        ggml_backend_sched_graph_compute(m->sched, step_gf);
        ggml_backend_tensor_get(step_logits, logits_buf.data(), 0, N * Vcb * sizeof(float));
        ggml_free(step_ctx);
        n_past++;
    }

    // ── Reverse delay pattern ───────────────────────────────────────────────
    int T_produced = (int)all_codes.size();
    T_raw = T_produced - N + 1;
    if (T_raw < 1) T_raw = 1;
    raw_codes.resize(T_raw * N);
    for (int t = 0; t < T_raw; t++)
        for (int c = 0; c < N; c++)
            raw_codes[t * N + c] = all_codes[t + c][c];

    ggml_backend_buffer_free(kv_buf);
    ggml_free(kv_ctx);
    return true;
}
