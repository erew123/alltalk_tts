#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

. ${SCRIPT_DIR=}/../variables.sh

DOCKER_TAG=latest

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --python-version)
      PYTHON_VERSION="$2"
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
    --github-repository)
      if [ -n "${GITHUB_REPOSITORY}" ] && ! [[ $GITHUB_REPOSITORY =~ ^--.* ]]; then
        GITHUB_REPOSITORY="$2"
        shift
      fi
      ;;
    *)
      echo "Unknown argument '$1'"
      exit 1
      ;;
  esac
  shift
done

PYTHON_SHORT_VERSION=${PYTHON_VERSION%.*}
PYTHON_VERSION_NO_DOT=${PYTHON_SHORT_VERSION//./}
if [[ -n $(find build -name "deepspeed-${DEEPSPEED_VERSION}*-cp${PYTHON_VERSION_NO_DOT}-cp${PYTHON_VERSION_NO_DOT}-*.whl") ]]
then
   echo "DeepSpeed was already built - skipping..."
   exit 0
fi

echo "Building DeepSpeed $DEEPSPEED_VERSION using python ${PYTHON_VERSION}"

rm -rf build # make sure to properly clean up
mkdir -p build

docker buildx \
  build \
  --build-arg GITHUB_REPOSITORY=$GITHUB_REPOSITORY \
  --build-arg DEEPSPEED_VERSION=$DEEPSPEED_VERSION \
  -t ${GITHUB_REPOSITORY}alltalk_deepspeed:${DOCKER_TAG} \
  .

docker run \
  --rm \
  -it \
  --gpus=all \
  --name deepspeed \
  -v $SCRIPT_DIR/build:/deepspeed \
  ${GITHUB_REPOSITORY}alltalk_deepspeed:${DOCKER_TAG}