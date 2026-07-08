# HiggsTTS.cpp

ggml port of [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b).

See [LICENSE-HIGGS](LICENSE-HIGGS).

## Build (HIP backend for example)

```bash
cd HiggsTTS.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DHIP_PLATFORM=amd
cmake --build build
```

Output: `build/bin/higgs_cli.exe`, `higgs_server.exe`, `higgs_quantize.exe`

## Model

Download GGUF from [NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf).

or you can convert the model yourself, see [here](convert_model)

## Usage

```bash
# CLI: WAV → WAV
higgs_cli --model model.gguf --ref-wav ref.wav --text "Hello world" --out out.wav

# Server
higgs_server --model model.gguf --ref-wav ref.wav --ref-text "reference transcript" --port 9989
```