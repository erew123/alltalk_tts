#!/bin/bash
# RVC Voice Training Script for macOS (Apple Silicon)
# Usage: ./train_rvc_voice.sh <model_name> <audio_folder> [epochs]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name> <audio_folder> [epochs]"
    echo ""
    echo "Arguments:"
    echo "  model_name    - Name for your voice model (e.g., 'my_voice')"
    echo "  audio_folder  - Path to folder containing WAV files"
    echo "  epochs        - Number of training epochs (default: 200)"
    echo ""
    echo "Example:"
    echo "  $0 john_voice /path/to/audio/files 300"
    exit 1
fi

MODEL_NAME="$1"
AUDIO_DIR="$2"
EPOCHS="${3:-200}"

# Convert to absolute path
if [[ "$AUDIO_DIR" != /* ]]; then
    AUDIO_DIR="$(cd "$AUDIO_DIR" 2>/dev/null && pwd)" || AUDIO_DIR=""
fi

# Validate audio directory
if [ -z "$AUDIO_DIR" ] || [ ! -d "$AUDIO_DIR" ]; then
    print_error "Audio directory not found: $2"
    exit 1
fi

# Count audio files
WAV_COUNT=$(find "$AUDIO_DIR" -name "*.wav" -o -name "*.mp3" -o -name "*.flac" 2>/dev/null | wc -l | tr -d ' ')
if [ "$WAV_COUNT" -eq 0 ]; then
    print_error "No audio files (wav/mp3/flac) found in: $AUDIO_DIR"
    exit 1
fi

echo -e "${GREEN}Found $WAV_COUNT audio files${NC}"

# Configuration
SAMPLE_RATE=48000
VERSION="v2"
BATCH_SIZE=4
SAVE_EVERY=10
F0_METHOD="rmvpe"  # Best quality pitch extraction
HOP_LENGTH=128

RVC_DIR="$SCRIPT_DIR/system/tts_engines/rvc"
LOGS_DIR="$RVC_DIR/logs/$MODEL_NAME"
PRETRAINED_G="$RVC_DIR/pretraineds/pretrained_v2/f0G48k.pth"
PRETRAINED_D="$RVC_DIR/pretraineds/pretrained_v2/f0D48k.pth"

# Check for pretrained models
print_step "Checking pretrained models..."

if [ ! -f "$PRETRAINED_G" ] || [ ! -f "$PRETRAINED_D" ]; then
    print_warning "Pretrained models not found. Downloading..."

    mkdir -p "$RVC_DIR/pretraineds/pretrained_v2"
    cd "$RVC_DIR/pretraineds/pretrained_v2"

    if [ ! -f "f0G48k.pth" ]; then
        echo "Downloading f0G48k.pth..."
        curl -L -o f0G48k.pth "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth"
    fi

    if [ ! -f "f0D48k.pth" ]; then
        echo "Downloading f0D48k.pth..."
        curl -L -o f0D48k.pth "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth"
    fi

    cd "$SCRIPT_DIR"
    echo -e "${GREEN}Pretrained models downloaded successfully${NC}"
else
    echo -e "${GREEN}Pretrained models found${NC}"
fi

# Activate conda environment
print_step "Activating conda environment..."
source "$SCRIPT_DIR/alltalk_environment/conda/bin/activate" "$SCRIPT_DIR/env"

# Set PYTHONPATH so rvc module can be found
export PYTHONPATH="$SCRIPT_DIR/system/tts_engines:$PYTHONPATH"

# Suppress PyTorch distributed warnings on macOS (not used for MPS training)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONWARNINGS="ignore::UserWarning"

# Change to RVC directory for training
cd "$RVC_DIR"

# Create logs directory
mkdir -p "$LOGS_DIR"

# Copy config file
CONFIG_SRC="$RVC_DIR/configs/v2/${SAMPLE_RATE}.json"
if [ -f "$CONFIG_SRC" ]; then
    cp "$CONFIG_SRC" "$LOGS_DIR/config.json"
    echo "Config copied to $LOGS_DIR/config.json"
else
    print_error "Config file not found: $CONFIG_SRC"
    exit 1
fi

# Step 1: Preprocessing
print_step "Step 1/6: Preprocessing audio files..."
python train/preprocess/preprocess.py \
    "$LOGS_DIR" \
    "$AUDIO_DIR" \
    "$SAMPLE_RATE" \
    3.0

# Check preprocessing output
if [ ! -d "$LOGS_DIR/1_16k_wavs" ] || [ -z "$(ls -A "$LOGS_DIR/1_16k_wavs" 2>/dev/null)" ]; then
    print_error "Preprocessing failed - no output files generated"
    exit 1
fi
echo -e "${GREEN}Preprocessing complete${NC}"

# Step 2: F0 (Pitch) Extraction
print_step "Step 2/6: Extracting pitch (F0) features..."
python train/extract/extract_f0_print.py \
    "$LOGS_DIR" \
    "$F0_METHOD" \
    "$HOP_LENGTH"

echo -e "${GREEN}Pitch extraction complete${NC}"

# Step 3: Feature Extraction
print_step "Step 3/6: Extracting voice features..."

# Determine device
if python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="mps"
    IS_HALF="False"
    echo "Using MPS (Apple Silicon) acceleration"
else
    DEVICE="cpu"
    IS_HALF="False"
    echo "Using CPU (no GPU acceleration)"
fi

python train/extract/extract_feature_print.py \
    "$DEVICE" \
    1 \
    0 \
    0 \
    "$LOGS_DIR" \
    "$VERSION" \
    "$IS_HALF" \
    hubert

echo -e "${GREEN}Feature extraction complete${NC}"

# Step 4: Prepare training files
print_step "Step 4/6: Preparing training files..."
python train/extract/preparing_files.py \
    "$LOGS_DIR" \
    "$VERSION" \
    "$SAMPLE_RATE"

echo -e "${GREEN}Training files prepared${NC}"

# Step 5: Training
print_step "Step 5/6: Starting training ($EPOCHS epochs)..."
echo "This may take a while depending on your dataset size and hardware."
echo "Training on: $DEVICE"
echo ""

python train/train.py \
    -se "$SAVE_EVERY" \
    -te "$EPOCHS" \
    -pg "$PRETRAINED_G" \
    -pd "$PRETRAINED_D" \
    -bs "$BATCH_SIZE" \
    -e "$LOGS_DIR" \
    -sr "$SAMPLE_RATE" \
    -sw "1" \
    -v "$VERSION" \
    -f0 1 \
    -l 1 \
    -c 0 \
    -od 1 \
    -ot 50 \
    -sg 0

echo -e "${GREEN}Training complete${NC}"

# Step 6: Create index
print_step "Step 6/6: Creating voice index..."
python train/process/extract_index.py \
    "$LOGS_DIR" \
    "$VERSION"

echo -e "${GREEN}Index creation complete${NC}"

# Summary
print_step "Training Complete!"
echo "Your trained model files are in:"
echo "  $LOGS_DIR"
echo ""
echo "Key files:"
ls -la "$LOGS_DIR"/*.pth 2>/dev/null | head -5 || echo "  (model files)"
ls -la "$LOGS_DIR"/*.index 2>/dev/null | head -2 || echo "  (index files)"
echo ""
echo "To use this voice in AllTalk:"
echo "  1. Copy the .pth model file to your RVC voices folder"
echo "  2. Copy the .index file alongside it"
echo "  3. Select the voice in AllTalk's RVC settings"
