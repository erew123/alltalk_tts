# AllTalk TTS - macOS Apple Silicon (M1/M2/M3) Installation Guide

This guide provides native ARM64 installation instructions for Apple Silicon Macs. This setup uses native ARM64 binaries for optimal performance and avoids Rosetta 2 emulation.

## Prerequisites

- macOS on Apple Silicon (M1, M2, M3, or newer)
- Terminal access
- Git installed

## Installation Steps

### 1. Clone the Repository

```bash
cd /path/to/your/preferred/directory
git clone -b alltalkbeta https://github.com/erew123/alltalk_tts
cd alltalk_tts
```

### 2. Create the Environment Directory

```bash
mkdir alltalk_environment
cd alltalk_environment
```

### 3. Download and Install Miniforge (ARM64 Native)

We use Miniforge instead of Miniconda because it provides native ARM64 builds via conda-forge:

```bash
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh -b -p $PWD/conda
rm Miniforge3-MacOSX-arm64.sh
```

### 4. Create the Conda Environment

```bash
./conda/bin/conda create --no-shortcuts -y -k --prefix ../env python=3.11.9
```

### 5. Activate the Environment

```bash
source ./conda/bin/activate ../env
```

### 6. Install PyTorch with MPS Support

PyTorch supports Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon:

```bash
conda install -y pytorch torchvision torchaudio -c pytorch
```

This installs PyTorch with MPS support, which uses Apple's Metal API for GPU acceleration.

### 7. Install Faiss and FFmpeg Dependencies

```bash
conda install -y faiss-cpu -c pytorch
conda install -y -c conda-forge libvorbis
```

### 8. Install Python Requirements

Navigate back to the alltalk_tts directory and install the macOS ARM64-specific requirements:

```bash
cd ..
pip install -r system/requirements/requirements_macos_arm64.txt
```

### 9. Install DeepSpeed (Optional)

DeepSpeed now supports Apple Silicon via MPS:

```bash
pip install deepspeed
```

Note: Some DeepSpeed features may have limited functionality on MPS compared to CUDA.

### 10. Install Parler TTS (Optional)

```bash
pip install -r system/requirements/requirements_parler.txt
```

### 11. Clean Up

```bash
./alltalk_environment/conda/bin/conda clean --all --force-pkgs-dirs -y
```

## Downloading Voice Models (Piper)

Piper TTS requires voice models to be downloaded. Models should be placed in the `models/piper/` directory.

### Create the Models Directory

```bash
mkdir -p models/piper
cd models/piper
```

### Download Default Voice (LJSpeech)

```bash
curl -L -o "en_US-ljspeech-high.onnx" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx?download=true"
curl -L -o "en_US-ljspeech-high.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json?download=true"
```

### Additional High-Quality Voices (from brycebeattie.com)

These are free, public domain voices optimized for Piper:

```bash
# US English Male Voices
curl -L -o "bryce.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/bryce.onnx"
curl -L -o "bryce.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/bryce.onnx.json"

curl -L -o "john.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/john.onnx"
curl -L -o "john.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/john.onnx.json"

curl -L -o "norman.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/norman.onnx"
curl -L -o "norman.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/norman.onnx.json"

# US English Female Voices
curl -L -o "kristin.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/kristin.onnx"
curl -L -o "kristin.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/kristin.onnx.json"

curl -L -o "lj-med.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/lj-med.onnx"
curl -L -o "lj-med.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/lj-med.onnx.json"

# UK English Female Voices
curl -L -o "cori-high.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/cori-high.onnx"
curl -L -o "cori-high.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/cori-high.onnx.json"

curl -L -o "cori-med.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/cori-med.onnx"
curl -L -o "cori-med.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/cori-med.onnx.json"

# Irish English Female Voice
curl -L -o "jenny.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/jenny.onnx"
curl -L -o "jenny.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/jenny.onnx.json"

# Multi-Speaker Models
curl -L -o "mv2.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/mv2.onnx"
curl -L -o "mv2.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/mv2.onnx.json"

curl -L -o "clean100.onnx" "https://sfo3.digitaloceanspaces.com/bkmdls/clean100.onnx"
curl -L -o "clean100.onnx.json" "https://sfo3.digitaloceanspaces.com/bkmdls/clean100.onnx.json"
```

### Available Voice Models

| Voice | Description | Size |
|-------|-------------|------|
| bryce.onnx | US English male | 61MB |
| john.onnx | US English male | 61MB |
| norman.onnx | US English male | 61MB |
| kristin.onnx | US English female | 61MB |
| lj-med.onnx | US English female (LJSpeech medium) | 61MB |
| en_US-ljspeech-high.onnx | US English female (LJSpeech high) | 109MB |
| cori-med.onnx | UK English female (medium) | 61MB |
| cori-high.onnx | UK English female (high quality) | 109MB |
| jenny.onnx | Irish English female | 61MB |
| mv2.onnx | Multi-speaker (16 voices) | 74MB |
| clean100.onnx | Multi-speaker (244 voices) | 74MB |

More Piper voices are available at: https://huggingface.co/rhasspy/piper-voices

## Setting Up RVC (Voice Conversion) - Optional

RVC (Retrieval-based Voice Conversion) allows you to transform TTS output using custom voice models. This requires additional setup on macOS ARM64.

### 1. Install fairseq (Python 3.11+ Compatible)

The official fairseq package has compatibility issues with Python 3.11+. Use this patched version:

```bash
cd /path/to/alltalk_tts
source ./alltalk_environment/conda/bin/activate ./env

# Downgrade pip temporarily for dependency resolution
pip install pip==24.0

# Install the patched fairseq with Python 3.11 support
pip install git+https://github.com/One-sixth/fairseq.git@main
```

### 2. Create RVC Model Directories

```bash
mkdir -p models/rvc_base
mkdir -p models/rvc_voices
```

### 3. Download RVC Base Models

Navigate to the RVC base models directory and download the required models:

```bash
cd models/rvc_base

# FCPE pitch extraction model (66MB)
curl -L -o "fcpe.pt" "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/fcpe.pt?download=true"

# Hubert base model (181MB)
curl -L -o "hubert_base.pt" "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/hubert_base.pt?download=true"

# RMVPE pitch extraction models
curl -L -o "rmvpe.pt" "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/rmvpe.pt?download=true"
curl -L -o "rmvpe.onnx" "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/rmvpe.onnx?download=true"

# ContentVec base model (181MB)
curl -L -o "contentvec_base.pt" "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/contentvec_base.pt?download=true"

cd ../..
```

### 4. Enable RVC in Configuration

Edit `confignew.json` and set `rvc_enabled` to `true`:

```json
"rvc_settings": {
    "rvc_enabled": true,
    ...
}
```

### 5. Add RVC Voice Models

Place your `.pth` RVC voice model files in the `models/rvc_voices/` directory. These are custom-trained voice models that can transform TTS output to sound like specific voices.

### RVC Base Models Reference

| Model | Size | Purpose |
|-------|------|---------|
| fcpe.pt | 66MB | FCPE pitch extraction |
| hubert_base.pt | 181MB | Hubert embeddings |
| rmvpe.pt | 173MB | RMVPE pitch extraction |
| rmvpe.onnx | 345MB | RMVPE ONNX variant |
| contentvec_base.pt | 181MB | ContentVec embeddings |

## Running AllTalk

### Using the Start/Stop Scripts

The easiest way to run AllTalk:

```bash
# Start the server
./start_alltalk.sh

# Stop the server
./stop_alltalk.sh
```

### Manual Start

```bash
cd /path/to/alltalk_tts
source ./alltalk_environment/conda/bin/activate ./env
python tts_server.py
```

### Accessing the Web Interface

Once the server is running, open your browser to: **http://localhost:7851/**

### API Endpoints

- Web interface: `http://localhost:7851/`
- List voices: `http://localhost:7851/api/voices`
- Current settings: `http://localhost:7851/api/currentsettings`

## Verifying MPS Support

To verify that PyTorch can use MPS acceleration:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

## Known Limitations

1. **DeepSpeed**: While DeepSpeed supports Apple Silicon, some advanced features designed for NVIDIA GPUs may not work or have reduced performance.

2. **CUDA-specific features**: Any features that specifically require CUDA will not work on macOS.

3. **piper-phonemize**: The separate piper-phonemize package is not available for ARM64 macOS. The piper-tts package bundles its own phonemizer.

4. **fairseq**: The official fairseq package has Python 3.11+ compatibility issues. Use the patched version from the One-sixth fork (see RVC setup section):
   ```bash
   pip install git+https://github.com/One-sixth/fairseq.git@main
   ```

5. **onnxruntime-gpu**: Not available on macOS. The regular `onnxruntime` package is used instead.

## Troubleshooting

### No Voices Found
Ensure voice models are placed in the correct directory:
```
alltalk_tts/models/piper/
```
Each voice requires both `.onnx` and `.onnx.json` files.

### MPS Not Available
If `torch.backends.mps.is_available()` returns False, ensure you're running macOS 12.3 or later.

### FFmpeg Not Found
If you see "ffmpeg not found" errors, install libvorbis:
```bash
source ./alltalk_environment/conda/bin/activate ./env
conda install -y -c conda-forge libvorbis
```

### Out of Memory Errors
MPS has different memory characteristics than CUDA. Try reducing batch sizes if you encounter memory issues.

### Performance Issues
For optimal performance:
- Close unnecessary applications
- Ensure your Mac is plugged in (not on battery)
- Use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` environment variable if memory management issues occur

### Port Already in Use
If port 7851 is already in use:
```bash
# Find and kill the process
lsof -ti:7851 | xargs kill -9
```

## Quick Reference

| Original (x86/CUDA) | macOS ARM64 |
|---------------------|-------------|
| Miniconda x86_64 | Miniforge ARM64 |
| onnxruntime-gpu | onnxruntime |
| CUDA | MPS (Metal) |
| DeepSpeed + CUDA | DeepSpeed + MPS |

## File Locations

| Item | Path |
|------|------|
| Conda environment | `alltalk_tts/alltalk_environment/conda` |
| Python environment | `alltalk_tts/env` |
| Voice models (Piper) | `alltalk_tts/models/piper/` |
| RVC base models | `alltalk_tts/models/rvc_base/` |
| RVC voice models | `alltalk_tts/models/rvc_voices/` |
| Configuration | `alltalk_tts/confignew.json` |
| macOS requirements | `alltalk_tts/system/requirements/requirements_macos_arm64.txt` |
| Start script | `alltalk_tts/start_alltalk.sh` |
| Stop script | `alltalk_tts/stop_alltalk.sh` |
