#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

. ${SCRIPT_DIR=}/../variables.sh

if [ -f $SCRIPT_DIR/build/environment-cu-${CUDA_VERSION}-cp-${PYTHON_VERSION}.yml ]; then
   echo "Environment file already exists - skipping..."
   exit 0
fi

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
    *)
      # Allow to pass arbitrary arguments to docker as well to be flexible:
      echo "Unknown argument '$1'"
      exit 1
      ;;
  esac
  shift
done

echo "Building conda environment using python ${PYTHON_VERSION} with CUDA ${CUDA_VERSION}"

rm -rf build # make sure to properly clean up
mkdir -p build

docker buildx \
  build \
  --build-arg CUDA_VERSION=$CUDA_VERSION \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --output=$SCRIPT_DIR/build \
  .