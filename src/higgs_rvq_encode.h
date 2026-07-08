// higgs_rvq_encode.h — CPU-side RVQ encoding for Higgs TTS

#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

// Find nearest codebook entry for each frame. x: [T, cdim], cb: [n_cb, cdim] (row-major).
static inline void rvq_nearest_neighbor(const float* x, int T, const float* cb, int n_cb, int cdim,
                                         std::vector<int32_t>& codes) {
    codes.resize(T);
    for (int t = 0; t < T; t++) {
        float best = INFINITY;
        int best_k = 0;
        for (int k = 0; k < n_cb; k++) {
            float d = 0;
            for (int c = 0; c < cdim; c++) {
                float diff = x[(size_t)t * cdim + c] - cb[(size_t)k * cdim + c];
                d += diff * diff;
            }
            if (d < best) { best = d; best_k = k; }
        }
        codes[t] = best_k;
    }
}

// Subtract cb[codes[t]] from residual at each frame.
// residual: [T, dim], cb: [n_cb, dim] (row-major).
static inline void rvq_subtract_cb(float* residual, int T, const float* cb, int dim,
                                    const std::vector<int32_t>& codes) {
    for (int t = 0; t < T; t++) {
        const float* entry = cb + (size_t)codes[t] * dim;
        for (int d = 0; d < dim; d++)
            residual[(size_t)t * dim + d] -= entry[d];
    }
}

// proj_out_subtract: residual -= proj_w @ cb_entry + proj_b at each frame.
// residual: [T, out_dim], cb_entry: [T, cdim] (from codebook).
// pw: [out_dim, cdim] col-major (i.e. ne0=cdim, ne1=out_dim in ggml).
// pb: [out_dim].
// Note: (cdim, out_dim) means proj_out weight stored as [cdim, out_dim] in memory.
static inline void rvq_subtract_proj(float* residual, int T, const float* cb_entry, int cdim,
                                      const float* pw, const float* pb, int out_dim) {
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < out_dim; o++) {
            float s = 0;
            for (int c = 0; c < cdim; c++)
                s += cb_entry[(size_t)t * cdim + c] * pw[(size_t)c + (size_t)o * cdim];
            residual[(size_t)t * out_dim + o] -= s + pb[o];
        }
    }
}
