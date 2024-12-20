#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

. ${SCRIPT_DIR=}/../variables.sh

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
    *)
      # Allow to pass arbitrary arguments to docker as well to be flexible:
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

CONDA_ENV=${SCRIPT_DIR=}/../conda/build/environment-cu-${CUDA_VERSION}-cp-${PYTHON_VERSION}.yml
if [ ! -f ${CONDA_ENV} ]; then
    echo "No conda environment found. Please run 'build-conda-env.sh' first."
    exit 1
fi

echo "Building DeepSpeed $DEEPSPEED_VERSION using python ${PYTHON_VERSION}"

rm -rf build # make sure to properly clean up
mkdir -p build
cp ${CONDA_ENV} ${SCRIPT_DIR=}/build

docker buildx \
  build \
  --build-arg DEEPSPEED_VERSION=$DEEPSPEED_VERSION \
  -t deepspeed:$DEEPSPEED_VERSION \
  .

docker run \
  --rm \
  -it \
  --gpus=all \
  --name deepspeed \
  -v $SCRIPT_DIR/build:/deepspeed \
  deepspeed:$DEEPSPEED_VERSION