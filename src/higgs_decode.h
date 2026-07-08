#pragma once

#include <cstdint>
#include <vector>

struct higgs_test_model;

/// Decode RVQ codes [T×N] → 24kHz mono PCM float.
bool higgs_decode(
    struct higgs_test_model * m,
    const int32_t * codes,          // [T*N] flat, t-major
    int             T_raw,
    int             N,
    std::vector<float> & pcm,       // output 24kHz float samples
    int           & T_pcm);
