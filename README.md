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

or you can convert the model yourself, see [here](https://github.com/Rafa00127/HiggsTTS.cpp/tree/main/convert_model)

## Usage

```bash
# CLI: WAV → WAV
higgs_cli --model model.gguf --ref-wav ref.wav --text "Hello world" --out out.wav

# CLI with emotion tags (optional, requires tokenizer.json)
higgs_cli --model model.gguf --ref-wav ref.wav --text "Hello world" --tokenizer tokenizer.json --out out.wav

# Server
higgs_server --model model.gguf --ref-wav ref.wav --ref-text "reference transcript" --port 9989

# Server with emotion tags (optional, requires tokenizer.json)
higgs_server --model model.gguf --ref-wav ref.wav --ref-text "reference transcript" --tokenizer tokenizer.json --port 9989
```

`--tokenizer` is optional. It enables recognition of special emotion / style / prosody tags
(e.g. `<|style:whispering|>`)
when they appear in the prompt text. Without it, the model falls back to the built-in
GGUF BPE tokenizer.

### Special Token Reference

See [PROMPTING.md](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/PROMPTING.md) for full details.

| Category | Examples |
|----------|----------|
| Emotion (21) | `emotion:elation` `emotion:anger` `emotion:sadness` `emotion:fear` ... |
| Style (3) | `style:singing` `style:shouting` `style:whispering` |
| SFX (9) | `sfx:laughter` `sfx:cough` `sfx:sigh` `sfx:sneeze` ... |
| Prosody (10) | `prosody:speed_slow` `prosody:pitch_high` `prosody:pause` ... |

Tags are used as `<|category:name|>` in the prompt, e.g. `<|emotion:elation|> Hello world!`

Prepend a tag to your text to control delivery: `<|emotion:elation|> Hello world!`