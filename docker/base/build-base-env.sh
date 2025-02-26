#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

. ${SCRIPT_DIR=}/../variables.sh

DOCKER_TAG=latest
DOCKER_REPOSITORY=erew123

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
    *)
      echo "Unknown argument '$1'"
      exit 1
      ;;
  esac
  shift
done

echo "Building conda environment using python ${PYTHON_VERSION} with CUDA ${CUDA_VERSION}"

docker buildx \
  build \
  --progress=plain \
  --build-arg CUDA_VERSION=$CUDA_VERSION \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  -t ${DOCKER_REPOSITORY}alltalk_tts_environment:${DOCKER_TAG} \
  .