#!/usr/bin/env bash

CUDA_VERSION=12.1.1
PYTHON_VERSION=3.11
PYTORCH_VERSION=2.2.1
DEEPSPEED_VERSION=0.16.1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

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
    --pytorch-version)
      PYTORCH_VERSION="$2"
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

PYTHON_VERSION_NO_DOT=${PYTHON_VERSION//./}
if [[ -n $(find build -name "deepspeed-${DEEPSPEED_VERSION}*-cp${PYTHON_VERSION_NO_DOT}-cp${PYTHON_VERSION_NO_DOT}-*.whl") ]]
then
   echo "DeepSpeed was already built - skipping..."
   exit 0
fi

echo "Building DeepSpeed $DEEPSPEED_VERSION for CUDA $CUDA_VERSION using python ${PYTHON_VERSION} with PyTorch ${PYTORCH_VERSION}"

rm -rf build # make sure to properly clean up - we only want 1 wheel at the time
mkdir -p build
docker buildx \
  build \
  --build-arg CUDA_VERSION=$CUDA_VERSION \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg PYTORCH_VERSION=$PYTORCH_VERSION \
  --build-arg DEEPSPEED_VERSION=$DEEPSPEED_VERSION \
  -t deepspeed:cu-$CUDA_VERSION-ds-$DEEPSPEED_VERSION \
  .

docker run \
  --rm \
  -it \
  --gpus=all \
  --name deepspeed \
  -v $SCRIPT_DIR/build:/deepspeed \
  deepspeed:cu-$CUDA_VERSION-ds-$DEEPSPEED_VERSION