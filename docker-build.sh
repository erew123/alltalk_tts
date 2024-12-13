#!/usr/bin/env bash

TTS_MODEL=xtts
DOCKER_TAG=latest

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
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

echo "Starting docker build process using TTS model '${TTS_MODEL}' and docker tag '${DOCKER_TAG}'"

docker buildx \
  build \
  --build-arg TTS_MODEL=$TTS_MODEL \
  -t alltalk_beta:${DOCKER_TAG} \
  .

echo "Docker build process finished"