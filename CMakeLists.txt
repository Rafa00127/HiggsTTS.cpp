cmake_minimum_required(VERSION 3.10...3.29)
project(HiggsTTS LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── ggml ────────────────────────────────────────────────────────────────────
option(HIGGS_GGML_STATIC "Statically link ggml into a single DLL" OFF)
if(NOT TARGET ggml)
    if(HIGGS_GGML_STATIC)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(ggml)
endif()

# ── core library ────────────────────────────────────────────────────────────
add_library(higgs_core STATIC
    src/core/audio_resample.cpp
    src/core/gguf_loader.cpp
    src/core/hf_tokenizer.cpp
)
target_include_directories(higgs_core PUBLIC src src/core ggml/src)
target_compile_features(higgs_core PUBLIC cxx_std_17)
target_link_libraries(higgs_core PUBLIC ggml ggml-base ggml-cpu)
if(TARGET ggml-hip)
    target_link_libraries(higgs_core PUBLIC ggml-hip)
endif()

# ── higgs_tts static library ────────────────────────────────────────────────
add_library(higgs_tts_lib STATIC
    src/higgs_tts.cpp
    src/higgs_prefill.cpp
    src/higgs_decode.cpp
    src/higgs_BAR.cpp
)
target_include_directories(higgs_tts_lib PUBLIC src)
target_compile_definitions(higgs_tts_lib PRIVATE _CRT_SECURE_NO_WARNINGS)
target_compile_features(higgs_tts_lib PUBLIC cxx_std_17)
target_link_libraries(higgs_tts_lib PUBLIC higgs_core)
if(NOT MSVC)
    target_compile_options(higgs_tts_lib PRIVATE -O3)
endif()

# ── higgs_cli ───────────────────────────────────────────────────────────────
add_executable(higgs_cli src/higgs_cli.cpp src/dr_wav.h)
target_compile_definitions(higgs_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
target_include_directories(higgs_cli PRIVATE src)
target_compile_features(higgs_cli PRIVATE cxx_std_17)
target_link_libraries(higgs_cli PRIVATE higgs_tts_lib)
if(NOT MSVC)
    target_compile_options(higgs_cli PRIVATE -O3)
endif()

# ── higgs_server ────────────────────────────────────────────────────────────
add_executable(higgs_server src/higgs_server.cpp src/dr_wav.h)
target_compile_definitions(higgs_server PRIVATE _CRT_SECURE_NO_WARNINGS)
target_include_directories(higgs_server PRIVATE src)
target_compile_features(higgs_server PRIVATE cxx_std_17)
target_link_libraries(higgs_server PRIVATE higgs_tts_lib)
if(NOT MSVC)
    target_compile_options(higgs_server PRIVATE -O3)
endif()

# ── higgs_tts DLL (C ABI for C# P/Invoke) ──────────────────────────────────
add_library(higgs_tts SHARED EXCLUDE_FROM_ALL src/higgs_c_api.cpp)
target_compile_definitions(higgs_tts PRIVATE HIGGS_TTS_SHARED _CRT_SECURE_NO_WARNINGS)
target_include_directories(higgs_tts PRIVATE src)
target_compile_features(higgs_tts PRIVATE cxx_std_17)
target_link_libraries(higgs_tts PRIVATE higgs_tts_lib)
if(NOT MSVC)
    target_compile_options(higgs_tts PRIVATE -O3)
endif()

# ── tokenizer_helper ─────────────────────────────────────────────────────────
add_executable(tokenizer_helper EXCLUDE_FROM_ALL src/tokenizer_helper.cpp src/dr_wav.h)
target_compile_definitions(tokenizer_helper PRIVATE _CRT_SECURE_NO_WARNINGS)
target_include_directories(tokenizer_helper PRIVATE src)
target_compile_features(tokenizer_helper PRIVATE cxx_std_17)
target_link_libraries(tokenizer_helper PRIVATE higgs_tts_lib)
if(NOT MSVC)
    target_compile_options(tokenizer_helper PRIVATE -O3)
endif()

# ── higgs_quantize ──────────────────────────────────────────────────────────
add_executable(higgs_quantize
    tools/higgs_quantize/main.cpp
    tools/higgs_quantize/common-ggml.cpp
)
target_include_directories(higgs_quantize PRIVATE tools/higgs_quantize ggml/src)
target_compile_features(higgs_quantize PRIVATE cxx_std_17)
target_link_libraries(higgs_quantize PRIVATE ggml ggml-base)
if(NOT MSVC)
    target_compile_options(higgs_quantize PRIVATE -O3)
endif()
