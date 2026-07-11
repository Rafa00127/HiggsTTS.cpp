// src/higgs_c_api.h
#pragma once
#include <stdint.h>

#ifdef _WIN32
  #ifdef HIGGS_TTS_SHARED
    #define HIGGS_API __declspec(dllexport)
  #else
    #define HIGGS_API __declspec(dllimport)
  #endif
#else
  #define HIGGS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct higgs_tts_handle higgs_tts_handle;

// 模型生命周期
HIGGS_API higgs_tts_handle* higgs_tts_load(const char* gguf_path);
HIGGS_API void              higgs_tts_free(higgs_tts_handle* h);

/// 可选：设置 HF tokenizer.json 路径（BPE 模式）。不调用则用 GGUF 内置 vocab。
HIGGS_API int higgs_tts_set_tokenizer(higgs_tts_handle* h, const char* tokenizer_path);

// ── 三个管线函数 ──
// 中间产物 (codes, PCM) 全由调用方管理，handle 不缓存任何状态。

/// 1. 参考音频 → ref codes
///    audio: 24kHz mono float32
///    out_codes: 调用者提供的 buffer, 大小 ≥ max_frames * 8
///    返回实际帧数, 失败返回 -1
HIGGS_API int higgs_tts_encode_ref(
    higgs_tts_handle* h,
    const float* audio, int n_samples,
    int32_t* out_codes);

/// 2. ref codes + 文本 → raw codes (AR 生成)
///    target_text: 要合成的目标文本
///    ref_text: 参考音频的转写文本（可为 nullptr）
///    in_codes / T_in: 上一步 encode_ref 的输出
///    out_codes: 调用者提供的 buffer, 大小 ≥ max_steps * 8
///    返回实际帧数, 失败返回 -1
HIGGS_API int higgs_tts_ar_generate(
    higgs_tts_handle* h,
    const char* target_text,
    const char* ref_text,
    int has_ref_text,
    const int32_t* in_codes, int T_in,
    float temperature, int seed,
    int32_t* out_codes);

/// 3. raw codes → PCM (24kHz mono float32)
///    out_pcm: 调用者提供的 buffer, 大小 ≥ max_samples
///    返回实际采样数, 失败返回 -1
HIGGS_API int higgs_tts_decode(
    higgs_tts_handle* h,
    const int32_t* codes, int T_raw,
    float* out_pcm);

#ifdef __cplusplus
}
#endif
