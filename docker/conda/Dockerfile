FROM continuumio/miniconda3:24.7.1-0 AS build

ARG CUDA_VERSION=12.4.1
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
    RESULT=$( { conda install -y \
      gcc_linux-64 \
      gxx_linux-64 \
      pytorch \
      pytorch-cuda=${CUDA_SHORT_VERSION} \
      nvidia/label/cuda-${CUDA_SHORT_VERSION}.0::cuda-toolkit \
      faiss-gpu=1.9.0 \
      libaio \
      conda-forge::ffmpeg=7.1.0 \
      conda-forge::portaudio=19.7.0 \
      -c anaconda \
      -c nvidia ; } 2>&1 )

    if [ ! -f /environment-cu-${CUDA_VERSION}-cp-${PYTHON_VERSION}.yml] ; then
      echo "Failed to install conda dependencies: $RESULT"
      exit 1
    fi

    conda clean -a && pip cache purge
EOR

##############################################################################
# Export conda environment:
##############################################################################
RUN <<EOR
  conda activate alltalk
  conda env export > /environment-cu-${CUDA_VERSION}-cp-${PYTHON_VERSION}.yml
EOR

##############################################################################
# Make environment file available to docker build output:
##############################################################################
FROM scratch
COPY --from=build /environment-*.yml /