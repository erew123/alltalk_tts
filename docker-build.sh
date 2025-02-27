#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

. ${SCRIPT_DIR=}/docker/variables.sh

TTS_MODEL=xtts
DOCKER_TAG=latest-${TTS_MODEL}
CLEAN=false
LOCAL_DEEPSPEED_BUILD=false
DOCKER_REPOSITORY=erew123

# Create required build directories if they don't exist
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
    --deepspeed-version)
      DEEPSPEED_VERSION="$2"
      shift
      ;;
    --tag)
      DOCKER_TAG="$2"
      shift
      ;;
    --docker-repository)
      if [ -n "$2" ] && ! [[ $2 =~ ^--.* ]]; then
        DOCKER_REPOSITORY="$2"
        shift
      else
        DOCKER_REPOSITORY=""
      fi
      ;;
    --local-deepspeed-build)
      LOCAL_DEEPSPEED_BUILD=true
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

# Append slash if missing:
DOCKER_REPOSITORY=$(echo "$DOCKER_REPOSITORY" | sed 's![^/]$!&/!')

if [ "$CLEAN" = true ]; then
  rm -rf ${SCRIPT_DIR=}/docker/deepspeed/build
  # Recreate directories after clean
  mkdir -p ${SCRIPT_DIR=}/docker/deepspeed/build
  echo "Cleaned and recreated build directories"  
fi

echo "Building base environment image"
$SCRIPT_DIR/docker/base/build-base-env.sh \
  --cuda-version ${CUDA_VERSION} \
  --python-version ${PYTHON_VERSION} \
  --docker-repository ${DOCKER_REPOSITORY} \
  --tag ${DOCKER_TAG}

if [ $? -ne 0 ]; then
  echo "Failed to build base environment image"
  exit 1
fi

if [ "$LOCAL_DEEPSPEED_BUILD" = true ]; then
  echo "Building DeepSpeed"
  $SCRIPT_DIR/docker/deepspeed/build-deepspeed.sh \
    --python-version ${PYTHON_VERSION} \
    --docker-repository ${DOCKER_REPOSITORY} \
    --tag ${DOCKER_TAG}

  if [ $? -ne 0 ]; then
    echo "Failed to build DeepSpeed"
    exit 1
  fi
fi

echo "Starting docker build process using TTS model '${TTS_MODEL}' and docker tag '${DOCKER_TAG}'"
echo "Building for CUDA $CUDA_VERSION using python ${PYTHON_VERSION}"

docker buildx \
  build \
  --progress=plain \
  --build-arg TTS_MODEL=$TTS_MODEL \
  --build-arg ALLTALK_DIR=$ALLTALK_DIR \
  --build-arg DEEPSPEED_VERSION=$DEEPSPEED_VERSION \
  --build-arg DOCKER_TAG=$DOCKER_TAG \
  --build-arg DOCKER_REPOSITORY=$DOCKER_REPOSITORY \
  -t ${DOCKER_REPOSITORY}alltalk_tts:${DOCKER_TAG} \
  .

echo "Docker build process finished. Use docker-start.sh to start the container."
