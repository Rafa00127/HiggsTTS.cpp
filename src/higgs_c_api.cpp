// higgs_c_api.cpp — C ABI wrapper for C# P/Invoke
#include "higgs_c_api.h"
#include "higgs_tts.h"
#include "higgs_prefill.h"
#include "higgs_BAR.h"
#include "higgs_decode.h"
#include "core/bpe.h"
#include "core/hf_tokenizer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct higgs_tts_handle {
    higgs_test_model model;
    std::string tokenizer_path;
    HFTokenizer hf_tok;
    bool hf_loaded = false;
};

extern "C" {

HIGGS_API higgs_tts_handle* higgs_tts_load(const char* gguf_path) {
    auto* h = new higgs_tts_handle();
    if (!higgs_test_load_vocab(gguf_path, &h->model)) {
        std::fprintf(stderr, "higgs_c_api: vocab load fail\n");
        delete h;
        return nullptr;
    }
    if (!higgs_test_load(gguf_path, &h->model)) {
        std::fprintf(stderr, "higgs_c_api: model load fail\n");
        delete h;
        return nullptr;
    }
    return h;
}

HIGGS_API int higgs_tts_set_tokenizer(higgs_tts_handle* h, const char* path) {
    if (!h || !path || !path[0]) return -1;
    h->tokenizer_path = path;
    h->hf_loaded = h->hf_tok.load(path);
    if (!h->hf_loaded) {
        std::fprintf(stderr, "higgs_c_api: tokenizer load fail: %s\n", path);
        return -1;
    }
    return 0;
}

HIGGS_API void higgs_tts_free(higgs_tts_handle* h) {
    if (!h) return;
    higgs_test_free(&h->model);
    delete h;
}

HIGGS_API int higgs_tts_encode_ref(higgs_tts_handle* h,
                                    const float* audio, int n_samples,
                                    int32_t* out_codes) {
    if (!h || !audio || n_samples < 1600 || !out_codes) return -1;

    std::vector<int32_t> codes;
    int T_frames = 0;
    if (!higgs_prefill_encode(&h->model, audio, n_samples, codes, T_frames))
        return -1;

    std::memcpy(out_codes, codes.data(), codes.size() * sizeof(int32_t));
    return T_frames;
}

HIGGS_API int higgs_tts_ar_generate(higgs_tts_handle* h,
                                     const char* target_text,
                                     const char* ref_text,
                                     int has_ref_text,
                                     const int32_t* in_codes, int T_in,
                                     float temperature, int seed,
                                     int32_t* out_codes) {
    if (!h || !target_text || !in_codes || T_in < 1 || !out_codes) return -1;

    const int N = 8;

    auto tokenize = [&](const std::string& t) -> std::vector<int32_t> {
        if (h->hf_loaded) return h->hf_tok.encode(t);
        return core_bpe::tokenize_simple(h->model.token_to_id, h->model.merge_rank, t.c_str());
    };

    auto target_tokens = tokenize(target_text);

    std::vector<int32_t> ref_text_tokens;
    if (has_ref_text && ref_text && ref_text[0]) {
        ref_text_tokens = tokenize(ref_text);
    }

    int L_audio = T_in + N - 1;
    auto prompt_ids = higgs_build_prompt(&h->model, target_tokens,
                                          ref_text_tokens, L_audio, true);

    std::vector<int32_t> raw_codes;
    int T_raw = 0;
    if (!higgs_backbone_ar(&h->model, in_codes, T_in, prompt_ids.data(),
                            (int)prompt_ids.size(), temperature, seed,
                            raw_codes, T_raw))
        return -1;

    std::memcpy(out_codes, raw_codes.data(), raw_codes.size() * sizeof(int32_t));
    return T_raw;
}

HIGGS_API int higgs_tts_decode(higgs_tts_handle* h,
                                const int32_t* codes, int T_raw,
                                float* out_pcm) {
    if (!h || !codes || T_raw < 1 || !out_pcm) return -1;

    const int N = 8;
    std::vector<float> pcm;
    int T_pcm = 0;
    if (!higgs_decode(&h->model, codes, T_raw, N, pcm, T_pcm))
        return -1;

    std::memcpy(out_pcm, pcm.data(), pcm.size() * sizeof(float));
    return T_pcm;
}

} // extern "C"
