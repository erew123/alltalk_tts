#!/usr/bin/env bash

TTS_MODEL=xtts
CUDA_VERSION=12.1.1
PYTHON_VERSION=3.11.9
PYTORCH_VERSION=2.2.1
DOCKER_TAG=latest

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cuda-version)
      CUDA_VERSION="$2"
      shift
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift
      ;;
    --pytorch-version)
      PYTORCH_VERSION="$2"
      shift
      ;;
    --tts_model)
      TTS_MODEL="$2"
      shift
      ;;
    --tag)
      DOCKER_TAG="$2"
      shift
      ;;
    *)
      printf '%s\n' "Invalid argument ($1)"
      exit 1
      ;;
  esac
  shift
done

echo "$PYTHON_VERSION -> ${PYTHON_VERSION%.*}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

PYTHON_MAJOR_MINOR=${PYTHON_VERSION%.*}
$SCRIPT_DIR/deepspeed/build-deepspeed.sh \
  --cuda-version ${CUDA_VERSION} \
  --python-version ${PYTHON_MAJOR_MINOR} \
  --pytorch-version ${PYTORCH_VERSION}

echo "Starting docker build process using TTS model '${TTS_MODEL}' and docker tag '${DOCKER_TAG}'"
echo "Building for CUDA $CUDA_VERSION using python ${PYTHON_VERSION} with PyTorch ${PYTORCH_VERSION}"


docker buildx \
  build \
  --progress=plain \
  --build-arg TTS_MODEL=$TTS_MODEL \
  --build-arg CUDA_VERSION=$CUDA_VERSION \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg PYTORCH_VERSION=$PYTORCH_VERSION \
  -t alltalk_beta:${DOCKER_TAG} \
  .

echo "Docker build process finished"