# Fish Speech Model Training Guide

This guide covers how to train custom voice models for Fish Speech to use with AllTalk TTS.

## Overview

Fish Speech uses a two-stage architecture:
1. **Semantic Token Model** - Converts text to semantic audio tokens
2. **Codec/Vocoder** - Converts semantic tokens to audio waveforms

For voice cloning, you typically only need to fine-tune the semantic model while using the pre-trained codec.

## Quick Start: Zero-Shot Voice Cloning (No Training Required)

Before training, try Fish Speech's zero-shot capabilities:

1. Place a reference audio file (10-30 seconds of clean speech) in `voices/`
2. Create a matching `.reference.txt` file with the transcript
3. Select the voice in AllTalk

Example:
```
voices/
├── my_voice.wav
└── my_voice.wav.reference.txt
```

If zero-shot quality isn't sufficient, proceed with fine-tuning below.

---

## Training Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space

### Software
- Python 3.10+
- CUDA 11.8 or 12.x
- Fish Speech package (already installed with AllTalk)

---

## Step 1: Prepare Training Data

### Audio Requirements
- **Format**: WAV (16-bit PCM recommended)
- **Sample Rate**: 44.1kHz (will be resampled automatically)
- **Quality**: Clean, noise-free recordings
- **Duration**:
  - Minimum: 10-15 minutes of speech
  - Recommended: 1-3 hours for best quality
  - Maximum: More data generally helps, but with diminishing returns

### Data Structure

Organize your data with audio files and matching transcripts:

```
training_data/
├── audio/
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
└── transcripts/
    ├── 001.txt
    ├── 002.txt
    └── ...
```

Or use a single manifest file (`filelist.txt`):
```
path/to/audio/001.wav|This is the transcript for the first audio file.
path/to/audio/002.wav|This is the transcript for the second audio file.
```

### Tips for Quality Data

1. **Clean Audio**: Remove background noise, music, and other speakers
2. **Consistent Volume**: Normalize audio to -23 LUFS
3. **Natural Speech**: Include varied intonation and emotion
4. **Accurate Transcripts**: Ensure transcripts exactly match spoken words
5. **Diverse Content**: Include questions, statements, exclamations

### Recommended Tools for Data Preparation

- **Whisper**: Auto-transcription (`whisperx` for word-level timestamps)
- **Audacity**: Audio editing and noise removal
- **FFmpeg**: Format conversion and audio processing

---

## Step 2: Install Training Dependencies

```bash
# Activate AllTalk environment
cd /path/to/alltalk_tts
source env/bin/activate  # Linux/Mac
# or: env\Scripts\activate  # Windows

# Install additional training dependencies
pip install tensorboard
pip install wandb  # Optional: for experiment tracking
```

---

## Step 3: Preprocess Training Data

### Using Fish Speech's Built-in Preprocessing

```bash
# Navigate to fish_speech package location
cd env/lib/python3.11/site-packages/fish_speech

# Create semantic tokens from audio
python tools/vqgan/extract_vq.py \
    --input-dir /path/to/training_data/audio \
    --output-dir /path/to/training_data/semantic_tokens \
    --checkpoint /path/to/alltalk_tts/models/fishspeech/openaudio-s1-mini/codec.pth

# Prepare text data
python tools/llama/build_dataset.py \
    --input /path/to/training_data \
    --output /path/to/training_data/dataset \
    --filelist /path/to/training_data/filelist.txt
```

### Alternative: Using the Fish Speech Web UI

Fish Speech provides a Gradio-based training interface:

```bash
python -m fish_speech.webui.train
```

This provides a user-friendly interface for:
- Data preprocessing
- Training configuration
- Training monitoring
- Model export

---

## Step 4: Configure Training

Create a training configuration file (`train_config.yaml`):

```yaml
# Model configuration
model:
  base_model: "openaudio-s1-mini"  # or "openaudio-s1" for larger model

# Data configuration
data:
  train_dataset: "/path/to/training_data/dataset"
  batch_size: 4  # Reduce if OOM errors
  num_workers: 4

# Training configuration
training:
  epochs: 100
  learning_rate: 1e-5
  warmup_steps: 1000
  gradient_accumulation_steps: 4
  save_every_n_epochs: 10

# LoRA configuration (recommended for fine-tuning)
lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Output
output_dir: "/path/to/output/my_voice_model"
```

---

## Step 5: Start Training

### Command Line Training

```bash
python -m fish_speech.train \
    --config train_config.yaml \
    --resume  # Optional: resume from checkpoint
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir /path/to/output/my_voice_model/logs

# View at http://localhost:6006
```

### Training Tips

1. **Start with LoRA**: Uses less VRAM and trains faster
2. **Watch the Loss**: Should decrease steadily, then plateau
3. **Listen to Samples**: Generate test samples every few epochs
4. **Don't Overtrain**: More epochs isn't always better

---

## Step 6: Export and Use Your Model

### Export the Fine-tuned Model

```bash
python -m fish_speech.tools.merge_lora \
    --base-model /path/to/openaudio-s1-mini \
    --lora-model /path/to/output/my_voice_model/checkpoint-best \
    --output /path/to/alltalk_tts/models/fishspeech/my_custom_model
```

### Configure AllTalk to Use Your Model

1. Copy your model to AllTalk's models directory:
```bash
cp -r /path/to/output/my_voice_model /path/to/alltalk_tts/models/fishspeech/
```

2. Update `available_models.json` to include your model:
```json
{
  "my_custom_model": {
    "model_path": "models/fishspeech/my_custom_model",
    "description": "My custom fine-tuned voice",
    "model_type": "openaudio"
  }
}
```

3. Restart AllTalk and select your model

---

## Alternative: Using LoRA Adapters

Instead of merging, you can use LoRA adapters directly:

1. Place LoRA weights in a subfolder:
```
models/fishspeech/openaudio-s1-mini/
├── model.pth
├── codec.pth
└── loras/
    └── my_voice/
        └── adapter.pth
```

2. Reference the LoRA in your voice configuration

---

## Troubleshooting

### Out of Memory (OOM) Errors

```yaml
# Reduce batch size
data:
  batch_size: 2

# Increase gradient accumulation
training:
  gradient_accumulation_steps: 8

# Use LoRA instead of full fine-tuning
lora:
  enabled: true
```

### Training Loss Not Decreasing

- Check data quality (clean audio, accurate transcripts)
- Try lower learning rate (1e-6)
- Ensure sufficient data quantity (>30 minutes)

### Generated Audio Sounds Robotic

- Train for more epochs
- Use more training data
- Check audio sample rate consistency

### Model Doesn't Capture Voice Characteristics

- Add more diverse training samples
- Ensure training audio has consistent speaker identity
- Try full fine-tuning instead of LoRA

---

## Advanced: Training from Scratch

For training a completely new model (not recommended for most users):

```bash
# Requires significant compute resources (8x A100 or similar)
python -m fish_speech.train \
    --config configs/train_from_scratch.yaml \
    --data /path/to/large_dataset  # 1000+ hours of audio
```

---

## Resources

- [Fish Speech GitHub](https://github.com/fishaudio/fish-speech)
- [Fish Speech Documentation](https://speech.fish.audio/docs)
- [OpenAudio Model Card](https://huggingface.co/fishaudio/openaudio-s1-mini)
- [Training FAQ](https://github.com/fishaudio/fish-speech/discussions)

---

## Quick Reference

| Task | Command |
|------|---------|
| Preprocess audio | `python tools/vqgan/extract_vq.py --input-dir ./audio` |
| Start training | `python -m fish_speech.train --config config.yaml` |
| Monitor training | `tensorboard --logdir ./logs` |
| Merge LoRA | `python -m fish_speech.tools.merge_lora --base-model ... --lora-model ...` |
| Test generation | `python -m fish_speech.inference --model ./my_model --text "Hello"` |
