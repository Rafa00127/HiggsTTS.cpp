#pragma once

#include <cstdint>
#include <vector>

struct higgs_test_model;

// ── Prompt building ───────────────────────────────────────────────────────────

/// Apply delay pattern: codes [T*N] → [(T+N-1)*N], BOC above, EOC below.
std::vector<int32_t> higgs_apply_delay_pattern(const int32_t* codes, int T, int N);

/// Build prompt token sequence for voice-clone mode.
/// target_tokens / ref_text_tokens: BPE tokenized text.
/// L_audio: delayed codes length (T + N - 1).
std::vector<int32_t> higgs_build_prompt(
    const struct higgs_test_model * m,
    const std::vector<int32_t>& target_tokens,
    const std::vector<int32_t>& ref_text_tokens,
    int L_audio,
    bool has_ref_audio);

// ── Backbone AR ───────────────────────────────────────────────────────────────

/// Trim trailing silence from PCM (default -60 dBFS threshold).
void higgs_trim_trailing_silence(std::vector<float>& pcm, float db_threshold = -60.0f);

/// Backbone AR: prefill 36L + autoregressive decode loop → raw RVQ codes.
/// Returns false on alloc/compute failure.
bool higgs_backbone_ar(
    struct higgs_test_model * m,
    const int32_t * codes,          // [T*8] prefill codes, t-major
    int              T_frames,      // prefill code frames
    const int32_t  * prompt_ids,    // build_prompt output
    int              L_prompt,      // prompt length
    float            temperature,   // sampling temperature
    int              seed,          // random seed (for reproducibility)
    std::vector<int32_t> & raw_codes, // output raw codes [T_raw*8] t-major
    int            & T_raw);        // output code frames
