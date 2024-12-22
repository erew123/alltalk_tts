#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

. ${SCRIPT_DIR=}/docker/variables.sh

TTS_MODEL=xtts
DOCKER_TAG=latest
CLEAN=false

# Create required build directories if they don't exist
mkdir -p ${SCRIPT_DIR=}/docker/conda/build
mkdir -p ${SCRIPT_DIR=}/docker/deepspeed/build
echo "Created build directories"

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
    --tts_model)
      TTS_MODEL="$2"
      shift
      ;;
    --tag)
      DOCKER_TAG="$2"
      shift
      ;;
    --clean)
      CLEAN=true
      ;;
    *)
      printf '%s\n' "Invalid argument ($1)"
      exit 1
      ;;
  esac
  shift
done

if [ "$CLEAN" = true ]; then
  rm -rf ${SCRIPT_DIR=}/docker/conda/build
  rm -rf ${SCRIPT_DIR=}/docker/deepspeed/build
  # Recreate directories after clean
  mkdir -p ${SCRIPT_DIR=}/docker/conda/build
  mkdir -p ${SCRIPT_DIR=}/docker/deepspeed/build
  echo "Cleaned and recreated build directories"  
fi

$SCRIPT_DIR/docker/conda/build-conda-env.sh \
  --cuda-version ${CUDA_VERSION} \
  --python-version ${PYTHON_VERSION}

$SCRIPT_DIR/docker/deepspeed/build-deepspeed.sh \
  --python-version ${PYTHON_VERSION}

echo "Starting docker build process using TTS model '${TTS_MODEL}' and docker tag '${DOCKER_TAG}'"
echo "Building for CUDA $CUDA_VERSION using python ${PYTHON_VERSION}"

docker buildx \
  build \
  --progress=plain \
  --build-arg TTS_MODEL=$TTS_MODEL \
  --build-arg ALLTALK_DIR=$ALLTALK_DIR \
  -t alltalk_beta:${DOCKER_TAG} \
  .

echo "Docker build process finished. Use docker-start.sh to start the container."
