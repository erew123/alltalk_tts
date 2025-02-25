#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR=}/docker/variables.sh

WITH_UI=true
DOCKER_TAG=latest
declare -a ADDITIONAL_ARGS=()

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift
      ;;
    --voices)
      VOICES="$2"
      shift
      ;;
    --rvc_voices)
      RVC_VOICES="$2"
      shift
      ;;
    --no_ui)
      WITH_UI=false
      ;;
    --tag)
      DOCKER_TAG="$2"
      shift
      ;;
    --github-repository)
      if [ -n "$2" ] && ! [[ $2 =~ ^--.* ]]; then
        GITHUB_REPOSITORY="$2"
        shift
      fi
      ;;
    *)
      # Allow to pass arbitrary arguments to docker as well to be flexible:
      ADDITIONAL_ARGS+=( $1 )
      ;;
  esac
  shift
done

# Compose docker arguments based on user input to the script:
declare -a DOCKER_ARGS=()

if [[ -n $CONFIG ]]; then
  # Mount the config file to docker_confignew.json:
  DOCKER_ARGS+=( -v ${CONFIG}:${ALLTALK_DIR}/docker_confignew.json )
fi

if [[ -n $VOICES ]]; then
  DOCKER_ARGS+=( -v ${VOICES}:${ALLTALK_DIR}/voices )
fi

if [[ -n $RVC_VOICES ]]; then
  DOCKER_ARGS+=( -v ${RVC_VOICES}:${ALLTALK_DIR}/models/rvc_voices )
fi

if [ "$WITH_UI" = true ] ; then
    DOCKER_ARGS+=( -p 7852:7852 )
fi

docker run \
  --rm \
  -it \
  -p 7851:7851 \
  --gpus=all \
  --name alltalk \
 "${DOCKER_ARGS[@]}" \
 "${ADDITIONAL_ARGS[@]}" \
  ${GITHUB_REPOSITORY}alltalk_tts:${DOCKER_TAG} &> /dev/stdout