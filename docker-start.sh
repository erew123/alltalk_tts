#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR=}/docker/variables.sh

WITH_UI=true
ENABLE_MULTI_ENGINE_MANAGER=false
MULTI_ENGINE_MANAGER_CONFIG=
DOCKER_TAG=latest-xtts
DOCKER_REPOSITORY=erew123
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
    --rvc-voices)
      RVC_VOICES="$2"
      shift
      ;;
    --no-ui)
      WITH_UI=false
      ;;
    --with-multi-engine-manager)
      ENABLE_MULTI_ENGINE_MANAGER=true
      if [ -n "$2" ] && ! [[ $2 =~ ^--.* ]]; then
        MULTI_ENGINE_MANAGER_CONFIG=$2
        shift
      fi
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
      # Allow to pass arbitrary arguments to docker as well to be flexible:
      ADDITIONAL_ARGS+=( $1 )
      ;;
  esac
  shift
done

# Append slash if missing:
DOCKER_REPOSITORY=$(echo "$DOCKER_REPOSITORY" | sed 's![^/]$!&/!')

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
    if [ "$ENABLE_MULTI_ENGINE_MANAGER" = true ] ; then
      DOCKER_ARGS+=( -p 7500:7500 )
    else
      DOCKER_ARGS+=( -p 7852:7852 )
    fi
fi

if [[ -n $MULTI_ENGINE_MANAGER_CONFIG ]]; then
  DOCKER_ARGS+=( -v ${MULTI_ENGINE_MANAGER_CONFIG}:${ALLTALK_DIR}/docker_mem_config.json )
fi

# Pass env variables:
DOCKER_ARGS+=( -e ENABLE_MULTI_ENGINE_MANAGER=${ENABLE_MULTI_ENGINE_MANAGER} )
DOCKER_ARGS+=( -e WITH_UI=${WITH_UI} )

docker run \
  --rm \
  -it \
  -p 7851:7851 \
  --gpus=all \
  --name alltalk \
 "${DOCKER_ARGS[@]}" \
 "${ADDITIONAL_ARGS[@]}" \
  ${DOCKER_REPOSITORY}alltalk_tts:${DOCKER_TAG} &> /dev/stdout