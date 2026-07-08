# higgs_quantize — GGUF re-quantization tool for Higgs TTS
# Usage: higgs_quantize input.gguf output-q6_k.gguf q6_k

set(TARGET higgs_quantize)
add_executable(${TARGET}
    main.cpp
    common-ggml.cpp
    common-ggml.h
)
target_include_directories(${TARGET} PRIVATE . ../../ggml/src ../../ggml/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE ggml ggml-base ${CMAKE_THREAD_LIBS_INIT})
if(NOT MSVC)
    target_compile_options(${TARGET} PRIVATE -O3)
endif()
