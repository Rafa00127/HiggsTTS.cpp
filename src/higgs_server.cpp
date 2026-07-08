// higgs_server.cpp — TCP TTS server for Higgs TTS.
// Prefill codes computed once per reference audio, reused for all requests.
// Protocol: 4-byte text_len(BE) + 4-byte temperature(BE float) + UTF-8 text
//           → 4-byte n_samples(BE) + float32 PCM @ 24kHz

#include "higgs_tts.h"
#include "higgs_prefill.h"
#include "higgs_decode.h"
#include "higgs_BAR.h"
#include "core/bpe.h"
#include "core/audio_resample.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#define SOCKET int
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket close
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        int n = send(fd, data + sent, len - sent, 0);
        if (n == SOCKET_ERROR) return false;
        sent += n;
    }
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
        g_ref_text_tokens = core_bpe::tokenize_simple(g_model.token_to_id, g_model.merge_rank, g_ref_text);
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
    auto target_tokens = core_bpe::tokenize_simple(g_model.token_to_id, g_model.merge_rank, text);

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
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            g_seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--port") && i + 1 < argc)
            g_port = atoi(argv[++i]);
    }
    if (!g_model_path || !g_ref_wav) {
        fprintf(stderr, "Usage: higgs_server --model <gguf> --ref-wav <wav> [--ref-text <str>] [--temperature <f>] [--seed <n>] [--port <n>]\n");
        return 1;
    }

    fprintf(stderr, "[higgs_server] loading...\n");
    if (!load_model()) return 1;
    if (!prepare_prefill()) return 1;
    fprintf(stderr, "[higgs_server] ready. ref codes cached (%d frames).\n", g_T_frames);

    server_loop();

    higgs_test_free(&g_model);
    return 0;
}
