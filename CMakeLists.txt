cmake_minimum_required(VERSION 3.10...3.29)
project(HiggsTTS LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── ggml ────────────────────────────────────────────────────────────────────
if(NOT TARGET ggml)
    add_subdirectory(ggml)
endif()

# ── core library ────────────────────────────────────────────────────────────
add_library(higgs_core STATIC
    src/core/audio_resample.cpp
    src/core/gguf_loader.cpp
)
target_include_directories(higgs_core PUBLIC src src/core ggml/src)
target_compile_features(higgs_core PUBLIC cxx_std_17)
target_link_libraries(higgs_core PUBLIC ggml ggml-base ggml-cpu)
if(TARGET ggml-hip)
    target_link_libraries(higgs_core PUBLIC ggml-hip)
endif()

# ── higgs_tts library ───────────────────────────────────────────────────────
add_library(higgs_tts STATIC
    src/higgs_tts.cpp
    src/higgs_prefill.cpp
    src/higgs_decode.cpp
    src/higgs_BAR.cpp
)
target_include_directories(higgs_tts PUBLIC src)
target_compile_definitions(higgs_tts PRIVATE _CRT_SECURE_NO_WARNINGS)
target_compile_features(higgs_tts PUBLIC cxx_std_17)
target_link_libraries(higgs_tts PUBLIC higgs_core)
if(NOT MSVC)
    target_compile_options(higgs_tts PRIVATE -O3)
endif()

# ── higgs_cli ───────────────────────────────────────────────────────────────
add_executable(higgs_cli src/higgs_cli.cpp src/dr_wav.h)
target_compile_definitions(higgs_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
target_include_directories(higgs_cli PRIVATE src)
target_compile_features(higgs_cli PRIVATE cxx_std_17)
target_link_libraries(higgs_cli PRIVATE higgs_tts)
if(NOT MSVC)
    target_compile_options(higgs_cli PRIVATE -O3)
endif()

# ── higgs_server ────────────────────────────────────────────────────────────
add_executable(higgs_server src/higgs_server.cpp src/dr_wav.h)
target_compile_definitions(higgs_server PRIVATE _CRT_SECURE_NO_WARNINGS)
target_include_directories(higgs_server PRIVATE src)
target_compile_features(higgs_server PRIVATE cxx_std_17)
target_link_libraries(higgs_server PRIVATE higgs_tts)
if(NOT MSVC)
    target_compile_options(higgs_server PRIVATE -O3)
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
