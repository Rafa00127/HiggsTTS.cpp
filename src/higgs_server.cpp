// higgs_server.cpp — TCP TTS server for Higgs TTS.
// Prefill codes computed once per reference audio, reused for all requests.
// Protocol: 4-byte text_len(BE) + 4-byte temperature(BE float) + UTF-8 text
//           → 4-byte n_samples(BE) + float32 PCM @ 24kHz

#include "higgs_tts.h"
#include "higgs_prefill.h"
#include "higgs_decode.h"
#include "higgs_BAR.h"
#include "core/bpe.h"
#include "core/hf_tokenizer.h"
#include "core/audio_resample.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#define SOCKET int
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket close
#define SD_SEND SHUT_WR
#endif

#include <atomic>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ── config ──────────────────────────────────────────────────────────────────
static const char*  g_model_path    = nullptr;
static const char*  g_ref_wav       = nullptr;
static const char*  g_ref_text      = nullptr;
static float        g_temperature   = 0.9f;
static int          g_seed          = 42;
static int          g_port          = 9989;
static const char*  g_tokenizer     = nullptr;
static HFTokenizer  g_hf_tok;
static bool         g_stream_mode  = false;
static int          g_max_actions  = 0;

// ── globals ─────────────────────────────────────────────────────────────────
static higgs_test_model            g_model;
static bool                        g_model_loaded = false;
static std::vector<int32_t>        g_ref_codes;     // cached prefill codes
static int                         g_T_frames = 0;
static std::vector<int32_t>        g_ref_text_tokens;
static std::vector<int32_t>        g_prompt_ids;
static int                         g_L_prompt = 0;
static std::mutex                  g_mutex;
static std::atomic<bool>           g_running{true};

// ── helpers ─────────────────────────────────────────────────────────────────
static void die(const char* msg) {
    fprintf(stderr, "[higgs_server] FATAL: %s\n", msg);
    exit(1);
}

static bool send_all(SOCKET fd, const char* data, int len) {
    int sent = 0;
    while (sent < len) {
        int n = send(fd, data + sent, len - sent,
#ifdef _WIN32
                     0
#else
                     MSG_NOSIGNAL
#endif
        );
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

// ── streaming constants ──────────────────────────────────────────────────────
static const int kCodebooks                   = 8;
static const int kSamplesPerFrame             = 960;
static const int kStreamLookaheadFrames       = 16;
static const int kStreamStepFrames            = 16;
static const int kTrailingSilenceStopSamples  = 2 * 24000;
static const float kSilenceRmsThreshold       = 0.001f;
static const size_t kMaxQueuedPcmFrames       = 4;
static const int kClientSendTimeoutMs         = 2000;

// ── streaming helpers ────────────────────────────────────────────────────────
static bool send_frame(SOCKET fd, uint8_t type, const void* data, uint32_t bytes) {
    uint32_t bytes_be = htonl(bytes);
    return send_all(fd, (const char*)&type, 1) &&
           send_all(fd, (const char*)&bytes_be, 4) &&
           (!bytes || send_all(fd, (const char*)data, (int)bytes));
}

static void configure_client_socket(SOCKET fd) {
#ifdef _WIN32
    DWORD timeout = kClientSendTimeoutMs;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
    struct timeval timeout = {
        kClientSendTimeoutMs / 1000,
        (kClientSendTimeoutMs % 1000) * 1000,
    };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
}

struct stream_state {
    SOCKET fd;
    std::vector<int32_t> codes;
    int decoded_frames = 0;
    std::vector<float> pending_silence;
    std::vector<std::vector<float>> pcm_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_ready;
    std::thread writer;
    bool producer_done = false;
    bool send_failed = false;
    bool stopped_for_silence = false;
};

static bool enqueue_pcm(stream_state* state, std::vector<float>&& pcm) {
    if (pcm.empty()) return true;
    std::lock_guard<std::mutex> lock(state->queue_mutex);
    if (state->send_failed || state->pcm_queue.size() >= kMaxQueuedPcmFrames) {
        state->send_failed = true;
        return false;
    }
    state->pcm_queue.push_back(std::move(pcm));
    state->queue_ready.notify_one();
    return true;
}

static void stream_writer(stream_state* state) {
    for (;;) {
        std::vector<float> pcm;
        {
            std::unique_lock<std::mutex> lock(state->queue_mutex);
            state->queue_ready.wait(lock, [&] {
                return state->send_failed || state->producer_done || !state->pcm_queue.empty();
            });
            if (state->send_failed || (state->producer_done && state->pcm_queue.empty())) return;
            pcm = std::move(state->pcm_queue.front());
            state->pcm_queue.erase(state->pcm_queue.begin());
        }
        if (!send_frame(state->fd, 1, pcm.data(), (uint32_t)(pcm.size() * sizeof(float)))) {
            std::lock_guard<std::mutex> lock(state->queue_mutex);
            state->send_failed = true;
            state->queue_ready.notify_one();
            return;
        }
    }
}

static bool stream_failed(stream_state* state) {
    std::lock_guard<std::mutex> lock(state->queue_mutex);
    return state->send_failed;
}

static bool is_quiet_pcm(const float* pcm, int samples) {
    if (samples <= 0) return true;
    double sum_squares = 0.0;
    for (int i = 0; i < samples; i++) sum_squares += pcm[i] * pcm[i];
    return std::sqrt(sum_squares / samples) < kSilenceRmsThreshold;
}

static bool flush_pending_silence(stream_state* state) {
    if (state->pending_silence.empty()) return true;
    std::vector<float> pcm;
    pcm.swap(state->pending_silence);
    return enqueue_pcm(state, std::move(pcm));
}

static bool stream_available_pcm(stream_state* state, bool final) {
    const int available = (int)state->codes.size() / kCodebooks;
    const int stable_end = final ? available : available - kStreamLookaheadFrames;
    if (stable_end <= state->decoded_frames ||
        (!final && stable_end - state->decoded_frames < kStreamStepFrames)) return true;

    const int window_start = std::max(0, state->decoded_frames - kStreamLookaheadFrames);
    std::vector<float> decoded;
    int decoded_samples = 0;
    if (!higgs_decode(&g_model, state->codes.data() + window_start * kCodebooks,
                      available - window_start, kCodebooks, decoded, decoded_samples)) {
        return false;
    }
    const int first = (state->decoded_frames - window_start) * kSamplesPerFrame;
    const int count = (stable_end - state->decoded_frames) * kSamplesPerFrame;
    if (decoded_samples < first + count) {
        return false;
    }
    state->decoded_frames = stable_end;
    const float* new_pcm = decoded.data() + first;
    if (is_quiet_pcm(new_pcm, count)) {
        state->pending_silence.insert(state->pending_silence.end(), new_pcm, new_pcm + count);
        if (!final && (int)state->pending_silence.size() >= kTrailingSilenceStopSamples) {
            state->stopped_for_silence = true;
            return false;
        }
        return true;
    }
    return flush_pending_silence(state) && enqueue_pcm(
        state, std::vector<float>(new_pcm, new_pcm + count));
}

static bool on_generated_frame(const int32_t* frame, void* user) {
    stream_state* state = (stream_state*)user;
    if (stream_failed(state)) return false;
    state->codes.insert(state->codes.end(), frame, frame + kCodebooks);
    return stream_available_pcm(state, false);
}

// ── streaming synthesis ──────────────────────────────────────────────────────
static bool synth_one_streaming(const char* text, float temperature, SOCKET fd) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto target_tokens = g_tokenizer ? g_hf_tok.encode(text)
        : core_bpe::tokenize_simple(g_model.token_to_id, g_model.merge_rank, text);

    int L_audio = g_T_frames + 7;
    auto prompt_ids = higgs_build_prompt(&g_model, target_tokens, g_ref_text_tokens, L_audio, true);
    int L_prompt = (int)prompt_ids.size();

    fprintf(stderr, "[higgs_server] synth: '%s' → %zu text tokens, prompt=%d, temperature=%.2f\n",
            text, target_tokens.size(), L_prompt, temperature);

    stream_state stream{fd};
    stream.writer = std::thread(stream_writer, &stream);
    bool completed = false;
    bool stopped_early = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        std::vector<int32_t> raw_codes;
        int T_raw = 0;
        if (!higgs_backbone_ar_stream(&g_model, g_ref_codes.data(), g_T_frames,
                                      prompt_ids.data(), L_prompt,
                                      temperature, g_seed, g_max_actions,
                                      raw_codes, T_raw,
                                      on_generated_frame, &stream, &stopped_early)) {
            fprintf(stderr, "[higgs_server] backbone AR failed\n");
            goto done;
        }
        if (!stream.stopped_for_silence && !stream_available_pcm(&stream, true)) goto done;
        if (stream_failed(&stream)) goto done;
        completed = stopped_early || stream.decoded_frames == T_raw;
    }
    if (!completed) {
        fprintf(stderr, "[higgs_server] stream frame count mismatch\n");
        goto done;
    }
done:
    {
        std::lock_guard<std::mutex> lock(stream.queue_mutex);
        stream.producer_done = true;
        stream.queue_ready.notify_one();
    }
    stream.writer.join();
    if (!completed || stream_failed(&stream)) return false;
    if (!send_frame(fd, 2, nullptr, 0)) return false;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[higgs_server] done: %.2f sec audio, %.0f ms%s\n",
            stream.decoded_frames * kSamplesPerFrame / 24000.0, ms,
            stream.stopped_for_silence ? " (trailing silence stopped)" : "");
    return true;
}

static bool load_model() {
    if (!higgs_test_load_vocab(g_model_path, &g_model)) {
        fprintf(stderr, "[higgs_server] vocab load failed\n");
        return false;
    }
    if (!higgs_test_load(g_model_path, &g_model)) {
        fprintf(stderr, "[higgs_server] model load failed\n");
        return false;
    }
    g_model_loaded = true;
    fprintf(stderr, "[higgs_server] model loaded: %s\n", g_model_path);
    return true;
}

// Read WAV → mono 24kHz float
static bool read_ref_audio(const char* path, std::vector<float>& samples) {
    unsigned int channels = 0, sr = 0;
    drwav_uint64 total = 0;
    float* data = drwav_open_file_and_read_pcm_frames_f32(path, &channels, &sr, &total, nullptr);
    if (!data) { fprintf(stderr, "[higgs_server] WAV read fail: %s\n", path); return false; }
    samples.resize(total);
    for (drwav_uint64 i = 0; i < total; i++) {
        float s = 0;
        for (unsigned int c = 0; c < channels; c++) s += data[i * channels + c];
        samples[i] = s / (float)channels;
    }
    drwav_free(data, nullptr);
    if ((int)sr != 24000)
        samples = core_audio::resample_polyphase(samples.data(), (int)samples.size(), (int)sr, 24000);
    return true;
}

// Pre-compute prefill codes + prompt (once, reused for all requests)
static bool prepare_prefill() {
    if (!g_model_loaded) return false;

    std::vector<float> wav_samples;
    if (!read_ref_audio(g_ref_wav, wav_samples)) return false;
    fprintf(stderr, "[higgs_server] ref audio: %zu samples @ 24kHz\n", wav_samples.size());

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!higgs_prefill_encode(&g_model, wav_samples.data(), (int)wav_samples.size(),
                               g_ref_codes, g_T_frames)) {
        fprintf(stderr, "[higgs_server] prefill encode failed\n");
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[higgs_server] prefill: %d frames × 8 codebooks (%.0f ms)\n", g_T_frames, prefill_ms);

    // Tokenize reference text (for prompt building)
    if (g_ref_text) {
        g_ref_text_tokens = g_tokenizer ? g_hf_tok.encode(g_ref_text)
            : core_bpe::tokenize_simple(g_model.token_to_id, g_model.merge_rank, g_ref_text);
        fprintf(stderr, "[higgs_server] ref text: %zu tokens\n", g_ref_text_tokens.size());
    }

    // Build prompt (shared — target text is empty placeholder, will vary per request)
    // We'll rebuild prompt per request since target text changes
    return true;
}

// Per-request synthesis
static bool synth_one(const char* text, float temperature, std::vector<float>& pcm) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Tokenize target text
    auto target_tokens = g_tokenizer ? g_hf_tok.encode(text)
        : core_bpe::tokenize_simple(g_model.token_to_id, g_model.merge_rank, text);

    // Build prompt with this target text
    int L_audio = g_T_frames + 7;  // T + N - 1, N=8
    auto prompt_ids = higgs_build_prompt(&g_model, target_tokens, g_ref_text_tokens, L_audio, true);
    int L_prompt = (int)prompt_ids.size();

    fprintf(stderr, "[higgs_server] synth: '%s' → %zu text tokens, prompt=%d, temperature=%.2f\n",
            text, target_tokens.size(), L_prompt, temperature);

    // Backbone AR + decode (both use m->sched, must be serialised)
    std::vector<int32_t> raw_codes;
    int T_raw = 0, T_pcm = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!higgs_backbone_ar(&g_model, g_ref_codes.data(), g_T_frames,
                                prompt_ids.data(), L_prompt,
                                temperature, g_seed, raw_codes, T_raw)) {
            fprintf(stderr, "[higgs_server] backbone AR failed\n");
            return false;
        }
        if (!higgs_decode(&g_model, raw_codes.data(), T_raw, 8, pcm, T_pcm)) {
            fprintf(stderr, "[higgs_server] decode failed\n");
            return false;
        }
    }

    higgs_trim_trailing_silence(pcm);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[higgs_server] done: %d PCM samples, %.2f sec audio, %.0f ms (RTF %.3f)\n",
            T_pcm, T_pcm / 24000.0, ms, ms / 1000.0 / (T_pcm / 24000.0));
    return true;
}

// ── TCP server ──────────────────────────────────────────────────────────────
static void handle_client(SOCKET client_fd) {
    configure_client_socket(client_fd);
    int32_t text_len_be = 0;
    int nr = recv(client_fd, (char*)&text_len_be, 4, MSG_WAITALL);
    if (nr != 4) { closesocket(client_fd); return; }
    int32_t text_len = ntohl(text_len_be);
    if (text_len <= 0 || text_len > 10000) { closesocket(client_fd); return; }

    int32_t temp_be = 0;
    nr = recv(client_fd, (char*)&temp_be, 4, MSG_WAITALL);
    if (nr != 4) { closesocket(client_fd); return; }
    int32_t temp_host = ntohl(temp_be);
    float temperature;
    memcpy(&temperature, &temp_host, sizeof(float));

    std::string text(text_len, '\0');
    nr = recv(client_fd, &text[0], text_len, MSG_WAITALL);
    if (nr != text_len) { closesocket(client_fd); return; }

    if (g_stream_mode) {
        // ── streaming protocol ────────────────────────────────────────────
        if (!synth_one_streaming(text.c_str(), temperature, client_fd)) {
            const char* error = "Higgs synthesis failed";
            send_frame(client_fd, 3, error, (uint32_t)strlen(error));
        }
    } else {
        // ── legacy one-shot protocol ──────────────────────────────────────
        std::vector<float> pcm;
        if (!synth_one(text.c_str(), temperature, pcm)) {
            int32_t err = htonl(-1);
            send_all(client_fd, (const char*)&err, 4);
            closesocket(client_fd);
            return;
        }
        int32_t ns_be = htonl((int32_t)pcm.size());
        send_all(client_fd, (const char*)&ns_be, 4);
        send_all(client_fd, (const char*)pcm.data(), (int)(pcm.size() * sizeof(float)));
    }
    shutdown(client_fd, SD_SEND);
    closesocket(client_fd);
}

static void server_loop() {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    SOCKET listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == INVALID_SOCKET) die("socket() failed");
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)g_port);
    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR)
        die("bind() failed");
    if (listen(listen_fd, 8) == SOCKET_ERROR)
        die("listen() failed");
    fprintf(stderr, "[higgs_server] listening on 127.0.0.1:%d\n", g_port);
    while (g_running) {
        SOCKET client = accept(listen_fd, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;
        std::thread(handle_client, client).detach();
    }
    closesocket(listen_fd);
#ifdef _WIN32
    WSACleanup();
#endif
}

// ── UTF-8 argv (Windows) ────────────────────────────────────────────────────
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
    int argc; LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wargv) return args;
    for (int i = 0; i < argc; i++) args.push_back(utf16_to_utf8(wargv[i]));
    LocalFree(wargv);
    return args;
}
#endif

// ── main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    setlocale(LC_ALL, ".UTF-8");
    auto utf8_args = get_utf8_argv();
    int nargs = (int)utf8_args.size();
    std::vector<const char*> cargs(nargs);
    for (int i = 0; i < nargs; i++) cargs[i] = utf8_args[i].c_str();
    argv = (char**)cargs.data();
    argc = nargs;
#endif
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            g_model_path = argv[++i];
        else if (!strcmp(argv[i], "--ref-wav") && i + 1 < argc)
            g_ref_wav = argv[++i];
        else if (!strcmp(argv[i], "--ref-text") && i + 1 < argc)
            g_ref_text = argv[++i];
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc)
            g_temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc)
            g_tokenizer = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            g_seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--port") && i + 1 < argc)
            g_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stream"))
            g_stream_mode = true;
        else if (!strcmp(argv[i], "--max-actions") && i + 1 < argc)
            g_max_actions = atoi(argv[++i]);
    }
    if (!g_model_path || !g_ref_wav) {
        fprintf(stderr, "Usage: higgs_server --model <gguf> --ref-wav <wav> [--ref-text <str>] [--tokenizer <json>] [--temperature <f>] [--seed <n>] [--port <n>] [--stream] [--max-actions <n>]\n");
        return 1;
    }
    if (g_max_actions < 0 || (g_max_actions > 0 && g_max_actions < kCodebooks)) {
        fprintf(stderr, "[higgs_server] --max-actions must be 0 or at least %d\n", kCodebooks);
        return 1;
    }
    if (g_stream_mode)
        fprintf(stderr, "[higgs_server] streaming mode enabled (lookahead=%d frames, step=%d)\n",
                kStreamLookaheadFrames, kStreamStepFrames);

    fprintf(stderr, "[higgs_server] loading...\n");
    if (!load_model()) return 1;
    if (g_tokenizer && g_hf_tok.load(g_tokenizer))
        fprintf(stderr, "[higgs_server] HF tokenizer loaded: %s\n", g_tokenizer);
    if (!prepare_prefill()) return 1;
    fprintf(stderr, "[higgs_server] ready. ref codes cached (%d frames).\n", g_T_frames);

    server_loop();

    higgs_test_free(&g_model);
    return 0;
}
