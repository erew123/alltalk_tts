# VibeVoice Setup Guide

This guide covers using **VibeVoice** with AllTalk TTS for long-form, multi-speaker speech synthesis (up to 90 minutes, up to 4 speakers per call).

## Overview

VibeVoice is Microsoft Research's open-source long-form TTS model. It uses continuous acoustic + semantic tokenizers at 7.5 Hz feeding a Qwen2.5 LLM with a diffusion head. Output is **24 kHz mono WAV**.

**Important:** Microsoft removed the VibeVoice-TTS inference code from their official repository in Sept 2025. AllTalk's VibeVoice engine uses the maintained community fork at [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice) (MIT). Weights remain available on HuggingFace.

## Hardware Requirements

| Device | Model | Memory | Speed |
|---|---|---|---|
| **NVIDIA CUDA** (recommended) | VibeVoice-1.5B, bf16 | ~5 GB VRAM | Real-time + |
| NVIDIA CUDA | VibeVoice-Large (~7B), bf16 | ~18 GB VRAM | Real-time |
| **Apple Silicon (MPS)** | VibeVoice-1.5B, fp32 | ~12 GB unified | ~0.5–1× RT |
| Apple Silicon | VibeVoice-Large | ~28 GB unified | not recommended |
| CPU | VibeVoice-1.5B, fp32 | ~12 GB RAM | very slow |

On CUDA, install `flash-attn` for best speed/quality. The engine falls back to `sdpa` automatically if it's missing.

## Installation

### 1. Install the VibeVoice package

The engine will auto-install on first use, but you can do it manually:

```bash
pip install git+https://github.com/vibevoice-community/VibeVoice.git
```

On CUDA, optionally:

```bash
pip install flash-attn --no-build-isolation
```

### 2. Download model weights

Weights auto-download from HuggingFace on first model load (~5–6 GB for 1.5B). To pre-download:

```bash
pip install "huggingface_hub[cli]"
hf download microsoft/VibeVoice-1.5b
```

For the Large variant:

```bash
hf download vibevoice-community/VibeVoice-Large-pt
```

Weights are cached at `~/.cache/huggingface/hub/`.

### 3. Add reference voice WAVs

VibeVoice clones a voice from a single reference WAV per speaker. Drop `.wav` files into:

```
system/tts_engines/vibevoice/voices/
```

Any `.wav` file in that folder becomes a usable voice. The basename (without `.wav`) is the voice name.

**Starter voices** from the community fork — clone them into your `voices/` folder:

```bash
git clone --depth 1 https://github.com/vibevoice-community/VibeVoice.git /tmp/vv
cp /tmp/vv/demo/voices/*.wav system/tts_engines/vibevoice/voices/
```

This gives you `en-Carter_man`, `en-Alice_woman`, `en-Frank_man`, `en-Maya_woman`, `en-Mary_woman_bgm` (musical), `in-Samuel_man` (Indian English), and three Mandarin voices.

**Custom voices:** record or trim 10–30 seconds of clean speech, save as `<name>.wav`.

### 4. Select VibeVoice in AllTalk

Open the AllTalk UI → TTS Engines → choose **vibevoice → VibeVoice-1.5b**. The model loads on selection; first load downloads weights if you skipped step 2.

## Usage

### Single-speaker

Standard AllTalk API call — pass any voice name from your `voices/` folder:

```
Hello, this is a single-voice generation.
```

### Multi-speaker dialogue

Prefix lines with `Speaker N:` (N = 1–4):

```
Speaker 1: Welcome back to the show.
Speaker 2: Thanks for having me on.
Speaker 1: Today we're talking about long-form synthesis.
Speaker 3: Mind if I jump in here?
```

Configure the **Speaker → Voice mapping** on the VibeVoice settings tab. Each speaker resolves to a reference WAV from your `voices/` folder.

### Tuning

Two knobs matter most:

- **CFG Scale** (default 1.3): higher = stricter adherence to the reference voice and script. Range 1.0–2.0 sane.
- **DDPM Inference Steps** (default 10): more = better fidelity, slower. 5 = fast preview, 10 = balanced.

## Training / Fine-Tuning

The community fork provides a fine-tuning recipe documented at:

> https://github.com/vibevoice-community/VibeVoice/blob/main/FINETUNING.md

It produces a LoRA checkpoint that AllTalk's engine **does not yet load**. To use a fine-tuned checkpoint, you'd currently need to either:

1. Merge the LoRA into the base weights using `peft` and point `model_variant` at the merged directory, or
2. Extend `api_manual_load_model` in `system/tts_engines/vibevoice/model_engine.py` to call `vibevoice.modular.lora_loading.load_lora_assets` after `from_pretrained`.

If/when you want LoRA support wired in, open an issue or extend that loader.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: vibevoice.modular...` | `pip install git+https://github.com/vibevoice-community/VibeVoice.git` |
| `Voice 'X' not found` | Add `voices/X.wav` to the engine folder |
| `flash_attention_2 failed` | Expected on Mac/CPU — engine falls back to sdpa automatically. On CUDA: `pip install flash-attn --no-build-isolation` |
| Very slow generation on Mac | MPS = fp32 only. Reduce DDPM steps to 5; try a shorter reference WAV |
| Robotic/garbled output | Bump CFG Scale to 1.5; use a longer (20–30s) cleaner reference WAV; check input punctuation is clean |
| Speaker swaps in dialogue | Make sure `Speaker N:` is at the start of each line and lines aren't being wrapped weirdly |

## License Notes

- **Code**: MIT (community fork) / Apache-style permissive
- **Weights**: Microsoft Research License — research use; review the model card on HuggingFace before commercial deployment

VibeVoice should not be used for impersonation, fraud, or disinformation. Disclose AI-generated audio when sharing.
