# HiggsTTS.cpp

ggml port of [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b).

See [LICENSE-HIGGS](LICENSE-HIGGS).

## Build

Personally I recommend Vulkan — it's fast and supports all platforms.
CUDA and HIP work too.

```bash
# Vulkan (recommended)
cmake -B build-vk -DGGML_VULKAN=ON
cmake --build build-vk --config Release -j

# CUDA
cmake -B build-cu -DGGML_CUDA=ON
cmake --build build-cu --config Release -j

# HIP (AMD)
cmake -B build-hip -DGGML_HIP=ON -DHIP_PLATFORM=amd
cmake --build build-hip --config Release -j
```

## Model

Download GGUF from [NeemaShioSe/HiggsTTS3.gguf](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf).

or you can convert the model yourself, see [here](https://github.com/Rafa00127/HiggsTTS.cpp/tree/main/convert_model)

## Usage

```bash
# CLI
higgs_cli --model higgs-v3-tts.gguf --ref-wav ref.wav --text "Hello world" --out out.wav

# CLI with emotion tags
higgs_cli --model higgs-v3-tts.gguf --ref-wav ref.wav --text "Hello world" --tokenizer tokenizer.json --out out.wav

# Server
higgs_server --model higgs-v3-tts.gguf --ref-wav ref.wav --ref-text "reference transcript" --port 9989

# Server with emotion tags
higgs_server --model higgs-v3-tts.gguf --ref-wav ref.wav --ref-text "reference transcript" --tokenizer tokenizer.json --port 9989

# Streaming mode
higgs_server --model higgs-v3-tts.gguf --ref-wav ref.wav --stream --port 9989

# Streaming with action cap
higgs_server --model higgs-v3-tts.gguf --ref-wav ref.wav --stream --max-actions 500
```

`--tokenizer` is optional. It enables recognition of special emotion / style / prosody tags
(e.g. `<|style:whispering|>`)
when they appear in the prompt text. Without it, the model falls back to the built-in
GGUF BPE tokenizer.

### Server options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *(required)* | Path to GGUF model file |
| `--ref-wav` | *(required)* | Reference audio WAV for voice cloning |
| `--ref-text` | — | Transcript of reference audio (improves quality) |
| `--tokenizer` | — | Path to HF `tokenizer.json` for emotion tag support |
| `--port` | `9989` | TCP listen port |
| `--temperature` | `0.9` | Sampling temperature |
| `--seed` | `42` | Random seed |
| `--stream` | off | Enable streaming protocol (framed PCM chunks instead of one-shot response) |
| `--max-actions` | `0` | Cap AR decode steps (0 = auto from text length). Streaming only. |

### Server protocol

**Request** (same for both modes):

```
[4B text_len BE][4B temperature BE float][UTF-8 text]
```

**Response — one-shot** (default, `--stream` not set):

```
[4B n_samples BE][float32 PCM @ 24kHz]
```

Error: `[4B int32 = -1 BE]`

**Response — streaming** (`--stream` set):

A sequence of frames, each: `[1B type][4B payload bytes BE][payload]`

| Type | Meaning |
|------|---------|
| `1` | float32 PCM chunk @ 24kHz |
| `2` | end of stream |
| `3` | UTF-8 error message |

See [python_gui/say.py](python_gui/say.py) for a Python client example.

### Quick demo with streaming

```bash
# Start the server in streaming mode (see Usage above), then:
pip install pyaudio
python python_gui/say.py "Hello world, this is a streaming TTS test."
```

The client connects, sends the text, and starts playing audio as soon as the first PCM
chunk arrives — no waiting for the full utterance to finish. `say.py` uses pyaudio for
gapless playback; pass `--no-stream` to fall back to the one-shot protocol.

> If you use an AI agent to assist with your work, try asking it to call `say.py` and say hello.

## GUI

### PyQt GUI (Python + TCP server)

A simple PyQt6 GUI that connects to `higgs_server` over TCP:
[python_gui/simple_tts_gui.py](python_gui/simple_tts_gui.py).

```bash
pip install pyqt6 numpy sounddevice
# Start the server first, then launch the GUI
python python_gui/simple_tts_gui.py
```

Configure model paths in Model Config, launch the server from the GUI,
then type text and click Synthesize.

### Windows GUI (WPF + C# bindings)

A native WPF GUI with direct C ABI bindings is provided in [CSharpBinding/](CSharpBinding/).
Runs local inference — no server needed.

```bash
# Build the native DLL
cmake --build build --target higgs_tts --config Release

# Build the .NET GUI
dotnet build CSharpBinding/HiggsTTS.net/HiggsTTSGUI/HiggsTTSGUI.csproj -c Release
```

Copy `higgs_tts.dll` and its dependencies (e.g. `ggml.dll`, `ggml-cpu.dll`, `ggml-vulkan.dll`)
from `build/bin/` to the GUI output directory before running.

The Tag dropdown gives quick access to all emotion / style / prosody tokens.
Built-in audio player supports play / pause / seek.

If you want to play some stupid tricks with your friend,
The GUI can also send synthesized speech to QQ as a voice message — see [HiggsTTSGUI/README.md](CSharpBinding/HiggsTTS.net/HiggsTTSGUI/README.md).


## Special Token Reference

See [PROMPTING.md](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/PROMPTING.md) for full details.

| Category | Examples |
|----------|----------|
| Emotion (21) | `emotion:elation` `emotion:anger` `emotion:sadness` `emotion:fear` ... |
| Style (3) | `style:singing` `style:shouting` `style:whispering` |
| SFX (9) | `sfx:laughter` `sfx:cough` `sfx:sigh` `sfx:sneeze` ... |
| Prosody (10) | `prosody:speed_slow` `prosody:pitch_high` `prosody:pause` ... |

Tags are used as `<|category:name|>` in the prompt, e.g. `<|emotion:elation|> Hello world!`

Prepend a tag to your text to control delivery: `<|emotion:elation|> Hello world!`