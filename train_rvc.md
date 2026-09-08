# RVC Voice Training Guide

This guide explains how to train custom RVC (Retrieval-based Voice Conversion) models for use with AllTalk TTS.

## Overview

RVC is a voice conversion technology that allows you to create custom voice models from audio samples. Unlike TTS, RVC converts one voice into another, preserving the original speech content while changing the voice characteristics.

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM) **OR**
- **Apple Silicon**: Mac with M1/M2/M3/M4 chip (MPS acceleration)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space for training data and checkpoints

### Software
- Python 3.10+
- PyTorch 2.5+ (required for MPS autocast support)
- Required packages: `faiss-cpu`, `pyworld`, `torchcrepe`, `parselmouth`, `librosa`

### Platform Notes

| Platform | GPU Support | FP16 Training | Notes |
|----------|-------------|---------------|-------|
| Windows/Linux | CUDA | Yes | Full support |
| macOS (Apple Silicon) | MPS | No (FP32 only) | PyTorch 2.5+ required |
| macOS (Intel) | CPU only | No | Not recommended |

## Training Pipeline

The RVC training process consists of 5 main steps:

```
Audio Files → Preprocessing → Feature Extraction → Training → Index Creation → Final Model
```

## Step 1: Prepare Audio Dataset

### Audio Requirements

| Requirement | Specification |
|-------------|---------------|
| Format | WAV (recommended), MP3, FLAC |
| Sample Rate | 32kHz, 40kHz, or 48kHz |
| Duration | 10-30 minutes total (optimal: 15-20 min) |
| Quality | Clean, noise-free recordings |
| Content | Speech only (no music, no background noise) |

### Best Practices for Dataset

1. **Clean Audio**: Remove background noise, music, and other speakers
2. **Consistent Volume**: Normalize audio levels
3. **Varied Content**: Include different emotions, tones, and speaking styles
4. **No Silence**: Trim long silences (the preprocessor handles short pauses)
5. **Single Speaker**: Each model should contain only one voice

### Dataset Structure

Place your audio files in a single directory:

```
training_data/
├── recording_001.wav
├── recording_002.wav
├── recording_003.wav
└── ...
```

## Step 2: Preprocessing

The preprocessor slices audio into segments and normalizes them.

### What It Does
- Applies high-pass filter (48Hz) to remove low-frequency noise
- Slices audio into manageable segments (default: 3 seconds with 0.3s overlap)
- Normalizes amplitude
- Creates 16kHz versions for feature extraction

### Output Directories
```
logs/{model_name}/
├── 0_gt_wavs/      # Ground truth wavs at target sample rate
└── 1_16k_wavs/     # 16kHz versions for feature extraction
```

### Preprocessing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 48000 | Target sample rate (32000, 40000, or 48000) |
| `percentage` | 3.0 | Segment length in seconds |

## Step 3: Feature Extraction

### F0 (Pitch) Extraction

F0 extraction captures the fundamental frequency (pitch) of the voice.

**Available Methods:**

| Method | Speed | Quality | GPU Required |
|--------|-------|---------|--------------|
| `pm` (Parselmouth) | Fast | Good | No |
| `harvest` | Slow | Excellent | No |
| `dio` | Fast | Good | No |
| `crepe` | Medium | Excellent | Yes |
| `rmvpe` | Medium | Best | Yes |

**Recommendation**: Use `rmvpe` for best quality, `pm` for fastest training.

### Embedding Extraction

Extracts voice features using a pre-trained model (HuBERT or similar).

**Output:**
- v1: 256-dimensional features
- v2: 768-dimensional features (recommended)

### Output Directories
```
logs/{model_name}/
├── 2a_f0/          # Coarse F0 features
├── 2b-f0nsf/       # Fine F0 features (for pitch guidance)
└── 3_feature768/   # Voice embeddings (v2)
```

## Step 4: Training

### Training Configuration

Key parameters in the config file:

```json
{
  "train": {
    "log_interval": 200,
    "epochs": 20000,
    "learning_rate": 1e-4,
    "batch_size": 4,
    "fp16_run": false
  }
}
```

### Training Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `epochs` | 200-500 | Number of training epochs |
| `batch_size` | 4-8 | Batch size (reduce if OOM) |
| `save_every_epoch` | 10-25 | Checkpoint frequency |
| `learning_rate` | 1e-4 | Initial learning rate |
| `fp16_run` | true/false | Mixed precision (faster, less VRAM) |

### Model Versions

| Version | Features | Recommended For |
|---------|----------|-----------------|
| v1 | 256-dim, simpler | Quick experiments |
| v2 | 768-dim, better quality | Production use |

### Pitch Guidance (F0)

- **With F0**: Better pitch accuracy, preserves intonation
- **Without F0**: Faster training, may lose some pitch nuances

**Recommendation**: Enable F0 for singing voices, optional for speech.

### Pretrained Models

RVC uses pretrained models as a starting point:

```
rvc/pretraineds/
├── pretrained_v1/
│   ├── f0G32k.pth, f0D32k.pth  # 32kHz with F0
│   ├── f0G40k.pth, f0D40k.pth  # 40kHz with F0
│   └── f0G48k.pth, f0D48k.pth  # 48kHz with F0
└── pretrained_v2/
    ├── f0G32k.pth, f0D32k.pth  # 32kHz with F0
    └── f0G48k.pth, f0D48k.pth  # 48kHz with F0
```

### Overtraining Detection

The training script includes overtraining detection:
- Monitors generator loss over epochs
- Stops if loss increases for a threshold number of epochs
- Helps prevent quality degradation

## Step 5: Index Creation

After training, an index file is created for voice retrieval:

```
logs/{model_name}/
├── trained_IVF{n}_Flat_nprobe_1_v2.index  # Trained index
└── added_IVF{n}_Flat_nprobe_1_v2.index    # Final index with all features
```

The index enables fast retrieval of similar voice features during inference.

## Training Workflow Summary

```bash
# 1. Set up experiment directory
export MODEL_NAME="my_voice"
export SAMPLE_RATE=48000
export VERSION="v2"

# 2. Preprocess audio
python rvc/train/preprocess/preprocess.py \
    logs/$MODEL_NAME \
    path/to/audio \
    $SAMPLE_RATE \
    3.0  # segment length

# 3. Extract F0 features
python rvc/train/extract/extract_f0_print.py \
    logs/$MODEL_NAME \
    rmvpe \
    128  # hop length

# 4. Extract voice embeddings
python rvc/train/extract/extract_feature_print.py \
    cuda:0 \
    1 \
    0 \
    0 \
    logs/$MODEL_NAME \
    $VERSION \
    True \
    hubert_base

# 5. Prepare training files
python rvc/train/extract/preparing_files.py \
    logs/$MODEL_NAME \
    $VERSION \
    $SAMPLE_RATE

# 6. Train the model
python rvc/train/train.py \
    --experiment_dir logs/$MODEL_NAME \
    --sample_rate $SAMPLE_RATE \
    --version $VERSION \
    --epochs 300

# 7. Create index
python rvc/train/process/extract_index.py \
    logs/$MODEL_NAME \
    $VERSION
```

## Output Files

After successful training, you'll have:

```
logs/{model_name}/
├── {model_name}_{epochs}e_{steps}s.pth  # Final model
├── added_IVF{n}_Flat_nprobe_1_v2.index  # Voice index
├── G_*.pth                               # Generator checkpoints
├── D_*.pth                               # Discriminator checkpoints
└── config.json                           # Training config
```

## Using Your Trained Model

1. Copy the `.pth` model file to your RVC voices directory
2. Copy the `.index` file alongside the model
3. Select the voice in AllTalk's RVC settings

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM (Out of Memory) | Reduce `batch_size`, enable `fp16_run` |
| Poor voice quality | More training data, more epochs |
| Unstable pitch | Enable F0 guidance, use `rmvpe` |
| Training too slow | Use GPU, enable fp16 |
| Index creation fails | Check GPU memory, reduce dataset size |

### Quality Tips

1. **More data is better**: 15-30 minutes of clean audio
2. **Diverse content**: Different emotions and speaking styles
3. **Clean recordings**: Remove noise, normalize volume
4. **Appropriate epochs**: Usually 200-500 epochs is sufficient
5. **Monitor loss**: Use TensorBoard to track training progress

### TensorBoard Monitoring

```bash
tensorboard --logdir logs/{model_name}
```

Monitor these metrics:
- `loss/g/total`: Generator total loss (should decrease)
- `loss/d/total`: Discriminator loss
- `loss/g/mel`: Mel spectrogram loss

## Sample Rates

| Rate | Use Case |
|------|----------|
| 32kHz | Lower quality, smaller models |
| 40kHz | Balanced (v1 only) |
| 48kHz | Highest quality (recommended) |

## Additional Resources

- [RVC GitHub Repository](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [HuBERT Model](https://huggingface.co/lj1995/VoiceConversionWebUI)
- [RMVPE Model](https://huggingface.co/lj1995/VoiceConversionWebUI)
