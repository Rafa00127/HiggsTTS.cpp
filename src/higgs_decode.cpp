// higgs_codes_test_main.cpp — full codec pipeline test
#include "higgs_tts.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
// #include "core/conv.h"  // not needed — convt1d decomposed manually below

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "higgs_decode.h"

bool higgs_decode(higgs_test_model* m, const int32_t* codes, int T_raw, int N,
                   std::vector<float>& pcm, int& T_pcm) {
    if (!m || !codes || T_raw < 1) return false;

    ggml_init_params ip = { m->compute_meta.size(), m->compute_meta.data(), true };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_tensor* quant_ids[16];
    ggml_tensor* decoded = nullptr;
    for (int c = 0; c < N; c++) {
        quant_ids[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_raw);
        ggml_set_input(quant_ids[c]);
        ggml_tensor* q = ggml_get_rows(ctx, m->quant[c].codebook, quant_ids[c]);
        q = ggml_mul_mat(ctx, m->quant[c].proj_out_w, q);
        if (m->quant[c].proj_out_b) q = ggml_add(ctx, q, m->quant[c].proj_out_b);
        decoded = decoded ? ggml_add(ctx, decoded, q) : q;
    }
    ggml_tensor* ac_in = ggml_mul_mat(ctx, m->fc2_w, decoded);
    if (m->fc2_b) ac_in = ggml_add(ctx, ac_in, m->fc2_b);
    ggml_set_name(ac_in, "ac_in");

    int C = (int)ac_in->ne[0], K = (int)m->conv1_w->ne[0], OC = (int)m->conv1_w->ne[2];
    ggml_tensor* im2col = ggml_im2col_rafa(ctx, ac_in, K, 1, 3, 1, m->conv1_w->type);
    ggml_tensor* w2d = ggml_reshape_2d(ctx, m->conv1_w, C*K, OC);
    ggml_tensor* y = ggml_mul_mat(ctx, im2col, w2d);
    if (m->conv1_b) { ggml_tensor* bb = ggml_reshape_2d(ctx, m->conv1_b, 1, OC);
                     y = ggml_add(ctx, y, bb); }

    // Snake1d: y = x + sin^2(alpha * x) / (alpha + 1e-9)
    ggml_tensor* snake_out = ggml_snake_1d(ctx, y, m->snake1_alpha);
    ggml_set_name(snake_out, "snake_out");

    // Block 0 ConvTranspose1d: [1024,91] → [512,728] (k=16, s=8, p=4)
    // Decomposed: mul_mat + col2im_1d — no causal trim (DAC uses symmetric pad)
    const int stride = 8, K_t = 16, OC_t = 512;
    ggml_tensor* x_cm = ggml_cont(ctx, ggml_transpose(ctx, snake_out));  // [91,1024] → [1024,91]
    // w_perm [IC, K*OC] = [1024, 8192], x_cm [IC, T_in] = [1024, 91]
    ggml_tensor* col = ggml_mul_mat(ctx, m->convt1_w, x_cm);              // [K*OC, T_in] = [8192, 91]
    ggml_tensor* t1 = ggml_col2im_1d(ctx, col, stride, OC_t, 0);         // [T_unpad, OC_t] = [736, 512]
    // DAC symmetric padding: k=2s, p=s/2 → crop s/2=4 from each end → T_out=T_in*s
    const int T_unpad = (int)t1->ne[0];
    const int crop = (stride + 1) / 2;  // ceil(s/2) for DAC symmetric pad
    t1 = ggml_view_2d(ctx, t1, T_unpad - 2*crop, OC_t, T_unpad * sizeof(float), crop * sizeof(float));
    t1 = ggml_cont(ctx, t1);  // [728, 512]
    // no transpose — [T, C] is ggml natural layout, downstream ops expect it
    if (m->convt1_b) {
        // bias [512] must broadcast along T dim: reshape → [512,1] → transpose → [1,512]
        ggml_tensor* bb = ggml_reshape_2d(ctx, m->convt1_b, m->convt1_b->ne[0], 1);
        bb = ggml_cont(ctx, ggml_transpose(ctx, bb));
        t1 = ggml_add(ctx, t1, bb);
    }
    ggml_set_name(t1, "convt1_out");

    // ---- ResUnit1 (dilation=1): snake1 → conv1(k=7) → snake2 → conv2(k=1) + skip ----
    // ① snake1
    ggml_tensor* r1 = ggml_snake_1d(ctx, t1, m->ru1_s1_alpha);  // [728, 512]
    ggml_set_name(r1, "ru1_snake1");
    // ② conv1(k=7, dil=1, pad=3): im2col_rafa + mul_mat
    ggml_tensor* r1_cm = ggml_cont(ctx, ggml_transpose(ctx, r1));                 // [512, 728]
    ggml_tensor* r1_im = ggml_im2col_rafa(ctx, r1_cm, 7, 1, 3, 1, m->ru1_c1_w->type);
    int ru1_C = (int)r1_cm->ne[0], ru1_OC = (int)m->ru1_c1_w->ne[2];
    ggml_tensor* r1_w2 = ggml_reshape_2d(ctx, m->ru1_c1_w, ru1_C * 7, ru1_OC);
    ggml_tensor* r1_c1 = ggml_mul_mat(ctx, r1_im, r1_w2);                         // [728, 512]
    if (m->ru1_c1_b) {
        ggml_tensor* bb = ggml_reshape_2d(ctx, m->ru1_c1_b, m->ru1_c1_b->ne[0], 1);
        bb = ggml_cont(ctx, ggml_transpose(ctx, bb));
        r1_c1 = ggml_add(ctx, r1_c1, bb);
    }
    // ③ snake2
    ggml_tensor* r1_s2 = ggml_snake_1d(ctx, r1_c1, m->ru1_s2_alpha);  // [728, 512]
    // ④ conv2(k=1): 1x1 conv = mul_mat, weight [1,512,512] → reshape_2d [512,512]
    ggml_tensor* r1_c2_cm = ggml_cont(ctx, ggml_transpose(ctx, r1_s2));          // [512, 728]
    ggml_tensor* r1_w1x1 = ggml_reshape_2d(ctx, m->ru1_c2_w, ru1_C, ru1_C);       // [512, 512] F16
    // std::printf("[type] r1_c2_cm=%s(%d) r1_w1x1=%s(%d) ru1_c2_w=%s(%d)\n",
    //             ggml_type_name(r1_c2_cm->type), (int)r1_c2_cm->type,
    //             ggml_type_name(r1_w1x1->type), (int)r1_w1x1->type,
    //             ggml_type_name(m->ru1_c2_w->type), (int)m->ru1_c2_w->type);
    if (r1_w1x1->type != GGML_TYPE_F32)
        r1_w1x1 = ggml_cpy(ctx, r1_w1x1, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ru1_C, ru1_C));
    ggml_tensor* r1_c2 = ggml_mul_mat(ctx, r1_c2_cm, r1_w1x1);                   // [728, 512]
    // r1_c2 = ggml_cont(ctx, ggml_transpose(ctx, r1_c2));                           // [728, 512]
    if (m->ru1_c2_b) {
        ggml_tensor* bb = ggml_reshape_2d(ctx, m->ru1_c2_b, m->ru1_c2_b->ne[0], 1);
        bb = ggml_cont(ctx, ggml_transpose(ctx, bb));
        r1_c2 = ggml_add(ctx, r1_c2, bb);
    }
    ggml_set_name(r1_c2, "ru1_conv2");
    // ⑤ skip connection
    ggml_tensor* ru1_out = ggml_add(ctx, t1, r1_c2);  // [728, 512]
    ggml_set_name(ru1_out, "ru1_out");

    // ---- ResUnit2 (dilation=3): snake1 → conv1(k=7,dil=3,pad=9) → snake2 → conv2(k=1) + skip ----
    // ① snake1
    ggml_tensor* r2 = ggml_snake_1d(ctx, ru1_out, m->ru2_s1_alpha);
    // ② conv1(k=7,dil=3,pad=9)
    ggml_tensor* r2_cm   = ggml_cont(ctx, ggml_transpose(ctx, r2));
    ggml_tensor* r2_im   = ggml_im2col_rafa(ctx, r2_cm, 7, 1, 9, 3, m->ru2_c1_w->type);
    int ru2_C = (int)r2_cm->ne[0], ru2_OC = (int)m->ru2_c1_w->ne[2];
    ggml_tensor* r2_w2   = ggml_reshape_2d(ctx, m->ru2_c1_w, ru2_C * 7, ru2_OC);
    ggml_tensor* r2_c1   = ggml_mul_mat(ctx, r2_im, r2_w2);
    if (m->ru2_c1_b) {
        ggml_tensor* bb = ggml_reshape_2d(ctx, m->ru2_c1_b, m->ru2_c1_b->ne[0], 1);
        bb = ggml_cont(ctx, ggml_transpose(ctx, bb));
        r2_c1 = ggml_add(ctx, r2_c1, bb);
    }
    // ③ snake2
    ggml_tensor* r2_s2 = ggml_snake_1d(ctx, r2_c1, m->ru2_s2_alpha);
    // ④ conv2(k=1)
    ggml_tensor* r2_c2_cm = ggml_cont(ctx, ggml_transpose(ctx, r2_s2));
    ggml_tensor* r2_w1x1  = ggml_reshape_2d(ctx, m->ru2_c2_w, ru2_C, ru2_C);
    if (r2_w1x1->type != GGML_TYPE_F32)
        r2_w1x1 = ggml_cpy(ctx, r2_w1x1, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ru2_C, ru2_C));
    ggml_tensor* r2_c2 = ggml_mul_mat(ctx, r2_c2_cm, r2_w1x1);
    if (m->ru2_c2_b) {
        ggml_tensor* bb = ggml_reshape_2d(ctx, m->ru2_c2_b, m->ru2_c2_b->ne[0], 1);
        bb = ggml_cont(ctx, ggml_transpose(ctx, bb));
        r2_c2 = ggml_add(ctx, r2_c2, bb);
    }
    // ⑤ skip
    ggml_tensor* ru2_out = ggml_add(ctx, ru1_out, r2_c2);
    ggml_set_name(ru2_out, "ru2_out");

    // ---- ResUnit3 (dilation=9): snake1 → conv1(k=7,dil=9,pad=27) → snake2 → conv2(k=1) + skip ----
    ggml_tensor* r3 = ggml_snake_1d(ctx, ru2_out, m->ru3_s1_alpha);
    ggml_tensor* r3_cm   = ggml_cont(ctx, ggml_transpose(ctx, r3));
    ggml_tensor* r3_im   = ggml_im2col_rafa(ctx, r3_cm, 7, 1, 27, 9, m->ru3_c1_w->type);
    int ru3_C = (int)r3_cm->ne[0], ru3_OC = (int)m->ru3_c1_w->ne[2];
    ggml_tensor* r3_w2   = ggml_reshape_2d(ctx, m->ru3_c1_w, ru3_C * 7, ru3_OC);
    ggml_tensor* r3_c1   = ggml_mul_mat(ctx, r3_im, r3_w2);
    if (m->ru3_c1_b) {
        ggml_tensor* bb = ggml_reshape_2d(ctx, m->ru3_c1_b, m->ru3_c1_b->ne[0], 1);
        bb = ggml_cont(ctx, ggml_transpose(ctx, bb));
        r3_c1 = ggml_add(ctx, r3_c1, bb);
    }
    ggml_tensor* r3_s2 = ggml_snake_1d(ctx, r3_c1, m->ru3_s2_alpha);
    ggml_tensor* r3_c2_cm = ggml_cont(ctx, ggml_transpose(ctx, r3_s2));
    ggml_tensor* r3_w1x1  = ggml_reshape_2d(ctx, m->ru3_c2_w, ru3_C, ru3_C);
    if (r3_w1x1->type != GGML_TYPE_F32)
        r3_w1x1 = ggml_cpy(ctx, r3_w1x1, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ru3_C, ru3_C));
    ggml_tensor* r3_c2 = ggml_mul_mat(ctx, r3_c2_cm, r3_w1x1);
    if (m->ru3_c2_b) {
        ggml_tensor* bb = ggml_reshape_2d(ctx, m->ru3_c2_b, m->ru3_c2_b->ne[0], 1);
        bb = ggml_cont(ctx, ggml_transpose(ctx, bb));
        r3_c2 = ggml_add(ctx, r3_c2, bb);
    }
    ggml_tensor* ru3_out = ggml_add(ctx, ru2_out, r3_c2);
    ggml_set_name(ru3_out, "ru3_out");

    // ---- Block 1 (stride=5): snake1 → ConvT1d(s=5) → 3×ResUnit ----
    // ① snake1 → ConvT1d(s=5, OC=b1_OCt)
    ggml_tensor* b1_s1 = ggml_snake_1d(ctx, ru3_out, m->b1_s1_alpha);
    int b1_stride = 5, b1_Kt = 10;
    int b1_OCt = (int)m->b1_convt_w->ne[1] / b1_Kt;
    ggml_tensor* b1_cm = ggml_cont(ctx, ggml_transpose(ctx, b1_s1));
    ggml_tensor* b1_col = ggml_mul_mat(ctx, m->b1_convt_w, b1_cm);
    ggml_tensor* b1_t = ggml_col2im_1d(ctx, b1_col, b1_stride, b1_OCt, 0);
    int b1_Tu = (int)b1_t->ne[0];
    int left_crop = (b1_stride + 1) / 2;  // ceil(s/2) for DAC symmetric pad
    b1_t = ggml_view_2d(ctx, b1_t, b1_Tu - b1_stride, b1_OCt, b1_Tu * sizeof(float), left_crop * sizeof(float));
    b1_t = ggml_cont(ctx, b1_t);
    if (m->b1_convt_b) { auto* bb = ggml_reshape_2d(ctx, m->b1_convt_b, m->b1_convt_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b1_t = ggml_add(ctx, b1_t, bb); }
    // ② ResUnit1 (dil=1)
    ggml_tensor* b1r1 = ggml_snake_1d(ctx, b1_t, m->b1_ru1_s1_a);
    ggml_tensor* b1r1_cm = ggml_cont(ctx, ggml_transpose(ctx, b1r1));
    int b1_C = (int)b1r1_cm->ne[0], b1_OC = (int)m->b1_ru1_c1_w->ne[2];
    ggml_tensor* b1r1_im = ggml_im2col_rafa(ctx, b1r1_cm, 7, 1, 3, 1, m->b1_ru1_c1_w->type);
    ggml_tensor* b1r1_w2 = ggml_reshape_2d(ctx, m->b1_ru1_c1_w, b1_C*7, b1_OC);
    ggml_tensor* b1r1_c1 = ggml_mul_mat(ctx, b1r1_im, b1r1_w2);
    if (m->b1_ru1_c1_b) { auto* bb = ggml_reshape_2d(ctx, m->b1_ru1_c1_b, m->b1_ru1_c1_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b1r1_c1 = ggml_add(ctx, b1r1_c1, bb); }
    ggml_tensor* b1r1_s2 = ggml_snake_1d(ctx, b1r1_c1, m->b1_ru1_s2_a);
    ggml_tensor* b1r1_c2cm = ggml_cont(ctx, ggml_transpose(ctx, b1r1_s2));
    ggml_tensor* b1r1_w1x1 = ggml_reshape_2d(ctx, m->b1_ru1_c2_w, b1_C, b1_C);
    if (b1r1_w1x1->type != GGML_TYPE_F32) b1r1_w1x1 = ggml_cpy(ctx, b1r1_w1x1, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b1_C, b1_C));
    ggml_tensor* b1r1_c2 = ggml_mul_mat(ctx, b1r1_c2cm, b1r1_w1x1);
    if (m->b1_ru1_c2_b) { auto* bb = ggml_reshape_2d(ctx, m->b1_ru1_c2_b, m->b1_ru1_c2_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b1r1_c2 = ggml_add(ctx, b1r1_c2, bb); }
    ggml_tensor* b1_ru1 = ggml_add(ctx, b1_t, b1r1_c2);
    // ResUnit2 (dil=3)
    ggml_tensor* b1r2 = ggml_snake_1d(ctx, b1_ru1, m->b1_ru2_s1_a);
    ggml_tensor* b1r2_cm = ggml_cont(ctx, ggml_transpose(ctx, b1r2));
    ggml_tensor* b1r2_im = ggml_im2col_rafa(ctx, b1r2_cm, 7, 1, 9, 3, m->b1_ru2_c1_w->type);
    ggml_tensor* b1r2_w2 = ggml_reshape_2d(ctx, m->b1_ru2_c1_w, b1_C*7, b1_OC);
    ggml_tensor* b1r2_c1 = ggml_mul_mat(ctx, b1r2_im, b1r2_w2);
    if (m->b1_ru2_c1_b) { auto* bb = ggml_reshape_2d(ctx, m->b1_ru2_c1_b, m->b1_ru2_c1_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b1r2_c1 = ggml_add(ctx, b1r2_c1, bb); }
    ggml_tensor* b1r2_s2 = ggml_snake_1d(ctx, b1r2_c1, m->b1_ru2_s2_a);
    ggml_tensor* b1r2_c2cm = ggml_cont(ctx, ggml_transpose(ctx, b1r2_s2));
    ggml_tensor* b1r2_w1x1 = ggml_reshape_2d(ctx, m->b1_ru2_c2_w, b1_C, b1_C);
    if (b1r2_w1x1->type != GGML_TYPE_F32) b1r2_w1x1 = ggml_cpy(ctx, b1r2_w1x1, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b1_C, b1_C));
    ggml_tensor* b1r2_c2 = ggml_mul_mat(ctx, b1r2_c2cm, b1r2_w1x1);
    if (m->b1_ru2_c2_b) { auto* bb = ggml_reshape_2d(ctx, m->b1_ru2_c2_b, m->b1_ru2_c2_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b1r2_c2 = ggml_add(ctx, b1r2_c2, bb); }
    ggml_tensor* b1_ru2 = ggml_add(ctx, b1_ru1, b1r2_c2);
    // ResUnit3 (dil=9)
    ggml_tensor* b1r3 = ggml_snake_1d(ctx, b1_ru2, m->b1_ru3_s1_a);
    ggml_tensor* b1r3_cm = ggml_cont(ctx, ggml_transpose(ctx, b1r3));
    ggml_tensor* b1r3_im = ggml_im2col_rafa(ctx, b1r3_cm, 7, 1, 27, 9, m->b1_ru3_c1_w->type);
    ggml_tensor* b1r3_w2 = ggml_reshape_2d(ctx, m->b1_ru3_c1_w, b1_C*7, b1_OC);
    ggml_tensor* b1r3_c1 = ggml_mul_mat(ctx, b1r3_im, b1r3_w2);
    if (m->b1_ru3_c1_b) { auto* bb = ggml_reshape_2d(ctx, m->b1_ru3_c1_b, m->b1_ru3_c1_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b1r3_c1 = ggml_add(ctx, b1r3_c1, bb); }
    ggml_tensor* b1r3_s2 = ggml_snake_1d(ctx, b1r3_c1, m->b1_ru3_s2_a);
    ggml_tensor* b1r3_c2cm = ggml_cont(ctx, ggml_transpose(ctx, b1r3_s2));
    ggml_tensor* b1r3_w1x1 = ggml_reshape_2d(ctx, m->b1_ru3_c2_w, b1_C, b1_C);
    if (b1r3_w1x1->type != GGML_TYPE_F32) b1r3_w1x1 = ggml_cpy(ctx, b1r3_w1x1, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b1_C, b1_C));
    ggml_tensor* b1r3_c2 = ggml_mul_mat(ctx, b1r3_c2cm, b1r3_w1x1);
    if (m->b1_ru3_c2_b) { auto* bb = ggml_reshape_2d(ctx, m->b1_ru3_c2_b, m->b1_ru3_c2_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b1r3_c2 = ggml_add(ctx, b1r3_c2, bb); }
    ggml_tensor* b1_out = ggml_add(ctx, b1_ru2, b1r3_c2);
    ggml_set_name(b1_out, "b1_out");

    // ---- Block 2 (stride=4) ----
    ggml_tensor* b2_s1 = ggml_snake_1d(ctx, b1_out, m->b2_s1_alpha);
    int b2_s = 4, b2_Kt = 8;
    int b2_OCt = (int)m->b2_convt_w->ne[1] / b2_Kt;
    ggml_tensor* b2_cm = ggml_cont(ctx, ggml_transpose(ctx, b2_s1));
    ggml_tensor* b2_col = ggml_mul_mat(ctx, m->b2_convt_w, b2_cm);
    ggml_tensor* b2_t = ggml_col2im_1d(ctx, b2_col, b2_s, b2_OCt, 0);
    int b2_Tu = (int)b2_t->ne[0];
    int b2_lc = (b2_s + 1) / 2;
    b2_t = ggml_view_2d(ctx, b2_t, b2_Tu - b2_s, b2_OCt, b2_Tu * sizeof(float), b2_lc * sizeof(float));
    b2_t = ggml_cont(ctx, b2_t);
    if (m->b2_convt_b) { auto* bb = ggml_reshape_2d(ctx, m->b2_convt_b, m->b2_convt_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b2_t = ggml_add(ctx, b2_t, bb); }
    // ResUnit1(dil=1) / ResUnit2(dil=3) / ResUnit3(dil=9)
    auto ru_block = [&](ggml_tensor* in, ggml_tensor* s1_a, ggml_tensor* c1_w, ggml_tensor* c1_b,
                         ggml_tensor* s2_a, ggml_tensor* c2_w, ggml_tensor* c2_b,
                         int dil, int pad, ggml_tensor** out_c2) {
        ggml_tensor* r = ggml_snake_1d(ctx, in, s1_a);
        ggml_tensor* r_cm = ggml_cont(ctx, ggml_transpose(ctx, r));
        int _C = (int)r_cm->ne[0], _OC = (int)c1_w->ne[2];
        ggml_tensor* r_im = ggml_im2col_rafa(ctx, r_cm, 7, 1, pad, dil, c1_w->type);
        ggml_tensor* r_w2 = ggml_reshape_2d(ctx, c1_w, _C*7, _OC);
        ggml_tensor* r_c1 = ggml_mul_mat(ctx, r_im, r_w2);
        if (c1_b) { auto* bb = ggml_reshape_2d(ctx, c1_b, c1_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); r_c1 = ggml_add(ctx, r_c1, bb); }
        ggml_tensor* r_s2 = ggml_snake_1d(ctx, r_c1, s2_a);
        ggml_tensor* r_c2cm = ggml_cont(ctx, ggml_transpose(ctx, r_s2));
        ggml_tensor* r_w1x1 = ggml_reshape_2d(ctx, c2_w, _C, _C);
        if (r_w1x1->type != GGML_TYPE_F32) r_w1x1 = ggml_cpy(ctx, r_w1x1, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, _C, _C));
        *out_c2 = ggml_mul_mat(ctx, r_c2cm, r_w1x1);
        if (c2_b) { auto* bb = ggml_reshape_2d(ctx, c2_b, c2_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); *out_c2 = ggml_add(ctx, *out_c2, bb); }
    };
    ggml_tensor *b2_c2_1, *b2_c2_2, *b2_c2_3;
    ru_block(b2_t, m->b2_ru1_s1_a, m->b2_ru1_c1_w, m->b2_ru1_c1_b, m->b2_ru1_s2_a, m->b2_ru1_c2_w, m->b2_ru1_c2_b, 1, 3, &b2_c2_1);
    ggml_tensor* b2_r1 = ggml_add(ctx, b2_t, b2_c2_1);
    ru_block(b2_r1, m->b2_ru2_s1_a, m->b2_ru2_c1_w, m->b2_ru2_c1_b, m->b2_ru2_s2_a, m->b2_ru2_c2_w, m->b2_ru2_c2_b, 3, 9, &b2_c2_2);
    ggml_tensor* b2_r2 = ggml_add(ctx, b2_r1, b2_c2_2);
    ru_block(b2_r2, m->b2_ru3_s1_a, m->b2_ru3_c1_w, m->b2_ru3_c1_b, m->b2_ru3_s2_a, m->b2_ru3_c2_w, m->b2_ru3_c2_b, 9, 27, &b2_c2_3);
    ggml_tensor* b2_out = ggml_add(ctx, b2_r2, b2_c2_3);
    ggml_set_name(b2_out, "b2_out");

    // ---- Block 3 (stride=2) ----
    ggml_tensor* b3_s1 = ggml_snake_1d(ctx, b2_out, m->b3_s1_alpha);
    int b3_s = 2, b3_Kt = 4;
    int b3_OCt = (int)m->b3_convt_w->ne[1] / b3_Kt;
    ggml_tensor* b3_cm = ggml_cont(ctx, ggml_transpose(ctx, b3_s1));
    ggml_tensor* b3_col = ggml_mul_mat(ctx, m->b3_convt_w, b3_cm);
    ggml_tensor* b3_t = ggml_col2im_1d(ctx, b3_col, b3_s, b3_OCt, 0);
    int b3_Tu = (int)b3_t->ne[0];
    int b3_lc = (b3_s + 1) / 2;
    b3_t = ggml_view_2d(ctx, b3_t, b3_Tu - b3_s, b3_OCt, b3_Tu * sizeof(float), b3_lc * sizeof(float));
    b3_t = ggml_cont(ctx, b3_t);
    if (m->b3_convt_b) { auto* bb = ggml_reshape_2d(ctx, m->b3_convt_b, m->b3_convt_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b3_t = ggml_add(ctx, b3_t, bb); }
    ggml_tensor *b3_c2_1, *b3_c2_2, *b3_c2_3;
    ru_block(b3_t, m->b3_ru1_s1_a, m->b3_ru1_c1_w, m->b3_ru1_c1_b, m->b3_ru1_s2_a, m->b3_ru1_c2_w, m->b3_ru1_c2_b, 1, 3, &b3_c2_1);
    ggml_tensor* b3_r1 = ggml_add(ctx, b3_t, b3_c2_1);
    ru_block(b3_r1, m->b3_ru2_s1_a, m->b3_ru2_c1_w, m->b3_ru2_c1_b, m->b3_ru2_s2_a, m->b3_ru2_c2_w, m->b3_ru2_c2_b, 3, 9, &b3_c2_2);
    ggml_tensor* b3_r2 = ggml_add(ctx, b3_r1, b3_c2_2);
    ru_block(b3_r2, m->b3_ru3_s1_a, m->b3_ru3_c1_w, m->b3_ru3_c1_b, m->b3_ru3_s2_a, m->b3_ru3_c2_w, m->b3_ru3_c2_b, 9, 27, &b3_c2_3);
    ggml_tensor* b3_out = ggml_add(ctx, b3_r2, b3_c2_3);
    ggml_set_name(b3_out, "b3_out");

    // ---- Block 4 (stride=3) ----
    ggml_tensor* b4_s1 = ggml_snake_1d(ctx, b3_out, m->b4_s1_alpha);
    int b4_s = 3, b4_Kt = 6;
    int b4_OCt = (int)m->b4_convt_w->ne[1] / b4_Kt;
    ggml_tensor* b4_cm = ggml_cont(ctx, ggml_transpose(ctx, b4_s1));
    ggml_tensor* b4_col = ggml_mul_mat(ctx, m->b4_convt_w, b4_cm);
    ggml_tensor* b4_t = ggml_col2im_1d(ctx, b4_col, b4_s, b4_OCt, 0);
    int b4_Tu = (int)b4_t->ne[0];
    int b4_lc = (b4_s + 1) / 2;
    b4_t = ggml_view_2d(ctx, b4_t, b4_Tu - b4_s, b4_OCt, b4_Tu * sizeof(float), b4_lc * sizeof(float));
    b4_t = ggml_cont(ctx, b4_t);
    if (m->b4_convt_b) { auto* bb = ggml_reshape_2d(ctx, m->b4_convt_b, m->b4_convt_b->ne[0], 1); bb = ggml_cont(ctx, ggml_transpose(ctx, bb)); b4_t = ggml_add(ctx, b4_t, bb); }
    ggml_tensor *b4_c2_1, *b4_c2_2, *b4_c2_3;
    ru_block(b4_t, m->b4_ru1_s1_a, m->b4_ru1_c1_w, m->b4_ru1_c1_b, m->b4_ru1_s2_a, m->b4_ru1_c2_w, m->b4_ru1_c2_b, 1, 3, &b4_c2_1);
    ggml_tensor* b4_r1 = ggml_add(ctx, b4_t, b4_c2_1);
    ru_block(b4_r1, m->b4_ru2_s1_a, m->b4_ru2_c1_w, m->b4_ru2_c1_b, m->b4_ru2_s2_a, m->b4_ru2_c2_w, m->b4_ru2_c2_b, 3, 9, &b4_c2_2);
    ggml_tensor* b4_r2 = ggml_add(ctx, b4_r1, b4_c2_2);
    ru_block(b4_r2, m->b4_ru3_s1_a, m->b4_ru3_c1_w, m->b4_ru3_c1_b, m->b4_ru3_s2_a, m->b4_ru3_c2_w, m->b4_ru3_c2_b, 9, 27, &b4_c2_3);
    ggml_tensor* b4_out = ggml_add(ctx, b4_r2, b4_c2_3);
    ggml_set_name(b4_out, "b4_out");

    // ---- Output: snake1 → conv2(32→1, k=7) → tanh ----
    ggml_tensor* out_s1 = ggml_snake_1d(ctx, b4_out, m->out_s1_alpha);
    ggml_tensor* out_cm = ggml_cont(ctx, ggml_transpose(ctx, out_s1));
    ggml_tensor* out_im = ggml_im2col_rafa(ctx, out_cm, 7, 1, 3, 1, m->out_conv2_w->type);
    int out_C = (int)out_cm->ne[0], out_OC = (int)m->out_conv2_w->ne[2];
    ggml_tensor* out_w2 = ggml_reshape_2d(ctx, m->out_conv2_w, out_C*7, out_OC);
    ggml_tensor* out_c2 = ggml_mul_mat(ctx, out_im, out_w2);
    if (m->out_conv2_b) { auto* bb = ggml_reshape_2d(ctx, m->out_conv2_b, 1, out_OC); out_c2 = ggml_add(ctx, out_c2, bb); }
    ggml_tensor* pcm_out = ggml_tanh(ctx, out_c2);
    ggml_set_name(pcm_out, "pcm");
    ggml_set_output(pcm_out);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, pcm_out);

    // Compute via scheduler (no gallocr)
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { std::fprintf(stderr, "alloc fail\n"); return false; }
    for (int c = 0; c < N; c++) {
        std::vector<int32_t> cb(T_raw);
        for (int t = 0; t < T_raw; t++) cb[t] = codes[t * N + c];
        ggml_backend_tensor_set(quant_ids[c], cb.data(), 0, T_raw * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(m->sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute fail\n"); return false;
    }

    // // dump im2col first 20 floats
    // {
    //     int n_im = (int)ggml_nelements(im2col);
    //     std::vector<uint8_t> raw(ggml_nbytes(im2col));
    //     ggml_backend_tensor_get(im2col, raw.data(), 0, ggml_nbytes(im2col));
    //     FILE* fim = std::fopen("dump_im2col.txt", "w");
    //     if (fim) {
    //         std::fprintf(fim, "im2col ne=[%lld,%lld,%lld,%lld] type=%s\n",
    //             im2col->ne[0], im2col->ne[1], im2col->ne[2], im2col->ne[3],
    //             ggml_type_name(im2col->type));
    //         int n = std::min(20, n_im);
    //         if (im2col->type == GGML_TYPE_F16) {
    //             auto* ptr = (ggml_fp16_t*)raw.data();
    //             for (int i = 0; i < n; i++)
    //                 std::fprintf(fim, "[%d] %.6f\n", i, ggml_fp16_to_fp32(ptr[i]));
    //         } else {
    //             auto* ptr = (float*)raw.data();
    //             for (int i = 0; i < n; i++)
    //                 std::fprintf(fim, "[%d] %.6f\n", i, ptr[i]);
    //         }
    //         std::fclose(fim);
    //     }
    // }
    //
    // // Dump
    // int n = (int)ggml_nelements(y);
    // ...
    // std::printf("Done. test_conv1d.txt\n");

    int n = (int)ggml_nelements(pcm_out);
    pcm.resize(n);
    ggml_backend_tensor_get(pcm_out, pcm.data(), 0, n * sizeof(float));
    T_pcm = n;

    ggml_free(ctx);
    return true;
}
