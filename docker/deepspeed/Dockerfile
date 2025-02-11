FROM continuumio/miniconda3:24.7.1-0

ARG DEEPSPEED_VERSION=0.16.2
ENV DEEPSPEED_VERSION=$DEEPSPEED_VERSION

SHELL ["/bin/bash", "-l", "-c"]
ENV SHELL=/bin/bash
ENV HOST=0.0.0.0
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CONDA_AUTO_UPDATE_CONDA="false"

##############################################################################
# Directories:
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

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
COPY build/environment-*.yml environment.yml
RUN <<EOR
    RESULT=$( { conda env create -f environment.yml ; } 2>&1 )

    if [ $? -ne 0 ] ; then
      echo "Failed to install conda dependencies: $RESULT"
      exit 1
    fi

    conda clean -a && pip cache purge
EOR

##############################################################################
# Install python dependencies
##############################################################################
ENV PIP_CACHE_DIR=/tmp/pip_cache
RUN <<EOR
  conda activate alltalk

  pip install \
      deepspeed-kernels \
      scikit-learn
EOR


##############################################################################
# Clone DeepSpeed sources
##############################################################################
RUN <<EOR
  git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
  cd ${STAGE_DIR}/DeepSpeed
  git checkout .
  git checkout "tags/v${DEEPSPEED_VERSION}" -b "v${DEEPSPEED_VERSION}"
EOR

##############################################################################
# Create DeepSpeed build file
##############################################################################
RUN <<EOR
  cat << EOF > build_deepspeed.sh
#!/usr/bin/env bash
source ~/.bashrc
mkdir -p /deepspeed

conda activate alltalk
cd ${STAGE_DIR}/DeepSpeed
DS_BUILD_OPS=1 python setup.py build_ext -j8 bdist_wheel
mv ${STAGE_DIR}/DeepSpeed/dist/*.whl /deepspeed/
EOF
EOR

RUN chmod +x /build_deepspeed.sh

##############################################################################
# Build DeepSpeed:
##############################################################################
ENTRYPOINT ["/build_deepspeed.sh"]