#!/usr/bin/env bash

ALLTALK_DIR="/alltalk"
WITH_UI=true
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
  alltalk_beta:latest &> /dev/stdout