// higgs_cli.cpp — End-to-end: WAV → prefill → backbone AR → decode → WAV
// Usage: higgs_cli --model <gguf> --ref-wav <wav> --text <str> [--ref-text <str>] [--temperature <f>] [--seed <n>] [--out <wav>]

#include "higgs_tts.h"
#include "higgs_prefill.h"
#include "higgs_decode.h"
#include "higgs_BAR.h"
#include "core/bpe.h"
#include "core/audio_resample.h"

#include "ggml.h"
#include "ggml-backend.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── Windows Unicode argv ─────────────────────────────────────────────────────
#ifdef _WIN32
static std::string utf16_to_utf8(const wchar_t* wstr) {
    if (!wstr || !*wstr) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string out(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &out[0], len, nullptr, nullptr);
    return out;
}

static std::vector<std::string> get_utf8_argv() {
    std::vector<std::string> args;
    int argc;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wargv) return args;
    for (int i = 0; i < argc; i++)
        args.push_back(utf16_to_utf8(wargv[i]));
    LocalFree(wargv);
    return args;
}
#endif

// ── WAV reader ────────────────────────────────────────────────────────────────
static bool read_wav_mono_24k(const char* path, std::vector<float>& samples) {
    unsigned int channels = 0, sr = 0;
    drwav_uint64 total = 0;
    float* data = drwav_open_file_and_read_pcm_frames_f32(path, &channels, &sr, &total, nullptr);
    if (!data) { fprintf(stderr, "wav read fail: %s\n", path); return false; }
    std::vector<float> mono(total);
    for (drwav_uint64 i = 0; i < total; i++) {
        mono[i] = 0.0f;
        for (unsigned int c = 0; c < channels; c++)
            mono[i] += data[i * channels + c];
        mono[i] /= (float)channels;
    }
    drwav_free(data, nullptr);
    if ((int)sr != 24000)
        samples = core_audio::resample_polyphase(mono.data(), (int)mono.size(), (int)sr, 24000);
    else
        samples = std::move(mono);
    return true;
}

// ── WAV writer ────────────────────────────────────────────────────────────────
static bool write_wav(const char* path, const std::vector<float>& pcm, int sr = 24000) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    int n = (int)pcm.size();
    int bits = 16, bps = bits / 8;
    int data_size = n * bps;
    auto w32 = [&](uint32_t v) { fwrite(&v, 1, 4, f); };
    auto w16 = [&](uint16_t v) { fwrite(&v, 1, 2, f); };
    fwrite("RIFF", 1, 4, f); w32(36 + data_size);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(1); w32(sr); w32(sr * bps); w16(bps); w16(bits);
    fwrite("data", 1, 4, f); w32(data_size);
    for (int i = 0; i < n; i++) {
        float v = pcm[i];
        if (v > 1.0f) v = 1.0f; if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t)(v * 32767.0f);
        fwrite(&s, 1, 2, f);
    }
    fclose(f);
    return true;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
#ifdef _WIN32
    auto utf8_args = get_utf8_argv();
    int nargs = (int)utf8_args.size();
    std::vector<const char*> cargs(nargs);
    for (int i = 0; i < nargs; i++) cargs[i] = utf8_args[i].c_str();
    argv = (char**)cargs.data();
    argc = nargs;
#endif

    const char* model_path = nullptr;
    const char* ref_wav    = nullptr;
    const char* text       = "I want a hero: an uncommon want";
    const char* ref_text   = nullptr;
    const char* out_wav    = "output.wav";
    float temperature      = 0.9f;
    int seed               = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            model_path = argv[++i];
        else if (!strcmp(argv[i], "--ref-wav") && i + 1 < argc)
            ref_wav = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc)
            text = argv[++i];
        else if (!strcmp(argv[i], "--ref-text") && i + 1 < argc)
            ref_text = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out_wav = argv[++i];
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc)
            temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
    }
    if (!model_path || !ref_wav) {
        fprintf(stderr, "Usage: higgs_BAR_test --model <gguf> --ref-wav <wav> [--text <str>] [--ref-text <str>] [--temperature <f>] [--seed <n>] [--out <wav>]\n");
        return 1;
    }

    // Load model
    higgs_test_model m;
    if (!higgs_test_load_vocab(model_path, &m)) { fprintf(stderr, "vocab fail\n"); return 1; }
    if (!higgs_test_load(model_path, &m))      { fprintf(stderr, "load fail\n"); return 1; }
    printf("Model loaded.\n");

    // ── Step 1: Prefill encode (WAV → RVQ codes) ─────────────────────────────
    std::vector<float> wav_samples;
    if (!read_wav_mono_24k(ref_wav, wav_samples)) { higgs_test_free(&m); return 1; }
    printf("Ref WAV: %zu samples @ 24kHz\n", wav_samples.size());

    std::vector<int32_t> codes;
    int T_frames = 0;
    if (!higgs_prefill_encode(&m, wav_samples.data(), (int)wav_samples.size(), codes, T_frames)) {
        fprintf(stderr, "Prefill encode failed\n");
        higgs_test_free(&m); return 1;
    }
    int N = 8;
    printf("Prefill: %d frames × %d codebooks\n", T_frames, N);

    // ── Step 2: Build prompt ─────────────────────────────────────────────────
    std::vector<int32_t> target_tokens = core_bpe::tokenize_simple(m.token_to_id, m.merge_rank, text);
    std::vector<int32_t> ref_text_tokens;
    if (ref_text)
        ref_text_tokens = core_bpe::tokenize_simple(m.token_to_id, m.merge_rank, ref_text);

    int L_audio = T_frames + N - 1;
    auto prompt_ids = higgs_build_prompt(&m, target_tokens, ref_text_tokens, L_audio, true);
    printf("Prompt: %zu tokens, L_audio=%d\n", prompt_ids.size(), L_audio);

    // ── Step 3: Backbone AR ──────────────────────────────────────────────────
    printf("Running backbone AR (temperature=%.2f, seed=%d)...\n", temperature, seed);
    auto t_ar_start = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> raw_codes;
    int T_raw = 0;
    if (!higgs_backbone_ar(&m, codes.data(), T_frames, prompt_ids.data(), (int)prompt_ids.size(),
                            temperature, seed, raw_codes, T_raw)) {
        fprintf(stderr, "Backbone AR failed\n");
        higgs_test_free(&m); return 1;
    }
    auto t_ar_end = std::chrono::high_resolution_clock::now();
    double ar_ms = std::chrono::duration<double, std::milli>(t_ar_end - t_ar_start).count();
    printf("Backbone AR: %d raw frames (%.0f ms)\n", T_raw, ar_ms);

    // ── Step 4: Decode ───────────────────────────────────────────────────────
    auto t_dec_start = std::chrono::high_resolution_clock::now();
    std::vector<float> pcm;
    int T_pcm = 0;
    if (!higgs_decode(&m, raw_codes.data(), T_raw, N, pcm, T_pcm)) {
        fprintf(stderr, "Decode failed\n");
        higgs_test_free(&m); return 1;
    }
    auto t_dec_end = std::chrono::high_resolution_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(t_dec_end - t_dec_start).count();
    double audio_sec = T_pcm / 24000.0;
    printf("Decode: %d PCM samples (%.2f sec) — %.0f ms\n", T_pcm, audio_sec, dec_ms);

    higgs_trim_trailing_silence(pcm);
    printf("Trim: %zu samples (%.2f sec)\n", pcm.size(), pcm.size() / 24000.0f);

    // ── Step 5: Save WAV ─────────────────────────────────────────────────────
    if (!write_wav(out_wav, pcm)) {
        fprintf(stderr, "Write WAV failed: %s\n", out_wav);
        higgs_test_free(&m); return 1;
    }

    printf("\n=== Timing ===\n");
    printf("Backbone AR: %8.0f ms\n", ar_ms);
    printf("Decode:      %8.0f ms\n", dec_ms);
    printf("Total gen:   %8.0f ms\n", ar_ms + dec_ms);
    printf("Audio:       %8.2f sec\n", audio_sec);
    printf("RTF:         %8.3f x\n", (ar_ms + dec_ms) / 1000.0 / audio_sec);

    printf("\nSaved: %s\n", out_wav);
    higgs_test_free(&m);
    return 0;
}
