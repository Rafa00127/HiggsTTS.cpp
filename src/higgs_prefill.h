#pragma once

#include <cstdint>
#include <vector>

struct higgs_test_model;

/// Encode reference audio (24 kHz mono f32) → RVQ codes [T × 8].
bool higgs_prefill_encode(
    struct higgs_test_model * m,
    const float  * audio,           // mono 24kHz f32
    int            n_samples,
    std::vector<int32_t> & codes,   // [T*8] flat, T-major (t0_q0..t0_q7, t1_q0..)
    int           & T_frames);
