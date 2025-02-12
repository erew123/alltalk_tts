FROM continuumio/miniconda3:24.7.1-0 AS build

ARG CUDA_VERSION=12.6.0
ENV CUDA_VERSION=$CUDA_VERSION

ARG PYTHON_VERSION=3.11.11
ENV PYTHON_VERSION=$PYTHON_VERSION

SHELL ["/bin/bash", "-l", "-c"]
ENV SHELL=/bin/bash
ENV HOST=0.0.0.0
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CONDA_AUTO_UPDATE_CONDA="false"

ENV ALLTALK_DIR=/opt/alltalk

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN <<EOR
    apt-get update
    apt-get upgrade -y
    apt-get install --no-install-recommends -y \
      espeak-ng \
      curl \
      wget \
      jq \
      vim

    apt-get clean && rm -rf /var/lib/apt/lists/*
EOR

##############################################################################
# Create a conda environment and install dependencies:
##############################################################################
RUN <<EOR
    CUDA_SHORT_VERSION=${CUDA_VERSION%.*}
    conda create -y -n "alltalk" -c conda-forge python=${PYTHON_VERSION}
    conda activate alltalk
    conda install -y \
      gcc_linux-64 \
      gxx_linux-64 \
      libaio \
      faiss-gpu=1.9.0 \
      cuda=${CUDA_VERSION} cuda-tools=${CUDA_VERSION} cuda-toolkit=${CUDA_VERSION} cuda-version=${CUDA_SHORT_VERSION} \
      cuda-command-line-tools=${CUDA_VERSION} cuda-compiler=${CUDA_VERSION} cuda-runtime=${CUDA_VERSION} \
      cuda-libraries=${CUDA_VERSION} cuda-libraries-dev=${CUDA_VERSION} cuda-libraries-static=${CUDA_VERSION} \
      cuda-cudart=${CUDA_SHORT_VERSION} cuda-cudart_linux-64=${CUDA_SHORT_VERSION} \
      cudnn=9.3 \
      py-cpuinfo=9 \
      conda-forge::ffmpeg=7.1.0 \
      conda-forge::portaudio=19.7.0 \
      -c pytorch \
      -c nvidia \
      -c anaconda

    if [ $? -ne 0 ] ; then
      echo "Failed to install conda dependencies: $RESULT"
      exit 1
    fi

    conda clean -a && pip cache purge
EOR

RUN <<EOR
  CUDA_SHORT_VERSION=${CUDA_VERSION%.*}
  CUDA_SHORT_VERSION_NO_DOT=${CUDA_SHORT_VERSION//./}

  conda activate alltalk
  mkdir -p ${ALLTALK_DIR}/pip_cache
  pip install --no-cache-dir --cache-dir=${ALLTALK_DIR}/pip_cache \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_SHORT_VERSION_NO_DOT}

  if [ $? -ne 0 ] ; then
    echo "Failed to install pip dependencies: $RESULT"
    exit 1
  fi
EOR