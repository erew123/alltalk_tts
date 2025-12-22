# Fish Speech Voice Training Guide

This guide explains how to train custom voice models for Fish Speech to use with AllTalk TTS.

## Overview

Fish Speech supports **zero-shot** and **few-shot** voice cloning, meaning you don't need to train a full model to clone a voice. Instead, you provide reference audio samples that the model uses to capture the voice characteristics.

### Two Approaches:

1. **Zero-Shot Cloning** (Recommended) - No training required, just provide reference audio
2. **Fine-Tuning** - Train a custom model on your voice data (advanced)

---

## Method 1: Zero-Shot Voice Cloning (No Training)

This is the easiest approach and works well for most use cases.

### Requirements
- 10-30 seconds of clear audio from the target voice
- A transcription of what was said in the audio

### Steps

1. **Prepare Reference Audio**
   - Record or obtain clear audio of the voice you want to clone
   - Format: WAV file, mono or stereo
   - Duration: 10-30 seconds (longer is generally better)
   - Quality: Clean audio, no background music or noise
   - Content: Natural speech with varied intonation

2. **Create Reference Text**
   - Create a text file with the **exact transcription** of the audio
   - Must match what was spoken word-for-word
   - Save with the same name as the audio file but with `.reference.txt` extension

3. **Place Files in Voices Folder**
   ```
   voices/
     my_custom_voice.wav
     my_custom_voice.reference.txt
   ```

4. **Use in AllTalk**
   - Select "my_custom_voice.wav" as the voice in AllTalk
   - The system will automatically use the reference text for conditioning

### Tips for Best Results

- **Audio Quality**: Use a good microphone, avoid echo and background noise
- **Speech Content**: Include varied emotions, pacing, and sentence types
- **Language Match**: Reference audio language should match your target language
- **Multiple Samples**: You can create multiple reference files for the same voice
  ```
  voices/narrator/
    sample1.wav + sample1.reference.txt
    sample2.wav + sample2.reference.txt
    sample3.wav + sample3.reference.txt
  ```

---

## Method 2: Fine-Tuning (Advanced)

For users who want to train a custom model with more control over voice characteristics.

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 12GB+ VRAM (24GB recommended)
- CUDA 11.8 or 12.x
- 1-10 hours of clean audio data

### Step 1: Install Fish Speech Training Environment

```bash
# Clone the Fish Speech repository
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .[train]
```

### Step 2: Prepare Training Data

1. **Collect Audio**
   - Gather 1-10 hours of clean audio from your target voice
   - WAV format, 44.1kHz or higher sample rate
   - Remove silence, music, and noise

2. **Create Transcriptions**
   - Transcribe all audio files
   - Use Whisper for automatic transcription:
     ```bash
     python tools/whisper_transcribe.py --input your_audio_folder/
     ```

3. **Organize Data Structure**
   ```
   data/
     my_voice/
       audio/
         001.wav
         002.wav
         ...
       transcripts.txt  # Format: filename|transcription
   ```

### Step 3: Preprocess Data

```bash
# Extract semantic tokens from audio
python fish_speech/models/dac/inference.py \
  -i data/my_voice/audio/ \
  -o data/my_voice/tokens/

# Prepare training dataset
python tools/prepare_dataset.py \
  --audio-dir data/my_voice/audio/ \
  --token-dir data/my_voice/tokens/ \
  --transcript data/my_voice/transcripts.txt \
  --output data/my_voice/dataset/
```

### Step 4: Fine-Tune the Model

```bash
# Start fine-tuning from base model
python fish_speech/train.py \
  --config configs/finetune.yaml \
  --data-dir data/my_voice/dataset/ \
  --base-model checkpoints/fish-speech-1.5/model.pth \
  --output-dir checkpoints/my_custom_model/ \
  --epochs 100 \
  --batch-size 4 \
  --learning-rate 1e-5
```

### Step 5: Export and Use

1. Copy your trained model to AllTalk:
   ```
   models/fishspeech/my_custom_model/
     model.pth
     config.json
     tokenizer.tiktoken
     special_tokens.json
     firefly-gan-vq-*.pth  # Copy from base model
   ```

2. Restart AllTalk and select your custom model

---

## LoRA Fine-Tuning (Lighter Weight)

For faster training with less VRAM:

```bash
python fish_speech/train_lora.py \
  --config configs/lora_finetune.yaml \
  --data-dir data/my_voice/dataset/ \
  --base-model checkpoints/fish-speech-1.5/model.pth \
  --output-dir checkpoints/my_lora/ \
  --lora-rank 16 \
  --epochs 50
```

---

## Training Tips

### Data Quality
- **Clean audio is crucial**: Remove all background noise, music, and artifacts
- **Consistent recording conditions**: Same microphone, room, and distance
- **Varied content**: Include different emotions, questions, statements
- **Natural speech**: Avoid overly theatrical or monotone delivery

### Hyperparameters
- **Learning rate**: Start with 1e-5, reduce if loss is unstable
- **Batch size**: 4-8 depending on VRAM
- **Epochs**: 50-200 depending on dataset size

### Common Issues

| Issue | Solution |
|-------|----------|
| Robotic output | More training data, lower learning rate |
| Doesn't match voice | More epochs, better quality reference audio |
| Unstable training | Reduce learning rate, check data quality |
| Out of memory | Reduce batch size, use LoRA |

---

## Resources

- [Fish Speech GitHub](https://github.com/fishaudio/fish-speech)
- [Fish Speech Documentation](https://speech.fish.audio/docs/)
- [Fish Audio Platform](https://fish.audio/) - Online voice cloning service
- [HuggingFace Models](https://huggingface.co/fishaudio)

---

## Quick Reference

### Minimum Requirements for Voice Cloning
| Method | Audio Length | Training Time | VRAM |
|--------|--------------|---------------|------|
| Zero-shot | 10-30 sec | None | 6GB+ |
| Fine-tune | 1-10 hours | 2-8 hours | 12GB+ |
| LoRA | 30 min - 2 hours | 30 min - 2 hours | 8GB+ |
