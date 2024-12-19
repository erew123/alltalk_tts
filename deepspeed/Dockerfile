ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11
ENV PYTHON_VERSION=$PYTHON_VERSION

ARG PYTORCH_VERSION=2.2.1
ENV PYTORCH_VERSION=$PYTORCH_VERSION

ARG DEEPSPEED_VERSION=0.16.1
ENV DEEPSPEED_VERSION=$DEEPSPEED_VERSION

ENV DEBIAN_FRONTEND=noninteractive

##############################################################################
# Directories:
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Installation/Basic Utilities
##############################################################################
SHELL ["/bin/bash", "-l", "-c"]
ENV SHELL=/bin/bash

RUN <<EOR
  apt-get update
  apt-get install --no-install-recommends -y \
    software-properties-common \
    ca-certificates

  add-apt-repository ppa:git-core/ppa -y # for latest git
  add-apt-repository ppa:deadsnakes/ppa -y # for python

  apt-get update
  apt-get upgrade -y
  apt-get install --no-install-recommends -y \
    build-essential \
    autotools-dev \
    pdsh \
    cmake \
    g++ \
    gcc \
    curl \
    wget \
    vim \
    unzip \
    llvm-dev \
    git \
    python${PYTHON_VERSION}-dev \
    libcupti-dev \
    libaio-dev
EOR

##############################################################################
# Python & pip
##############################################################################
RUN <<EOR
  # Correct symlinks to use the proper python version:
  PYTHON_MAJOR_VERSION=${PYTHON_VERSION%%.*}
  rm -f /usr/bin/python${PYTHON_MAJOR_VERSION}
  ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python${PYTHON_MAJOR_VERSION}
  ln -s /usr/bin/python3 /usr/bin/python

  curl -O https://bootstrap.pypa.io/pip/3.7/get-pip.py
  python get-pip.py
  rm get-pip.py
  pip install --upgrade pip
EOR

# Minimal dependencies needed to build deepspeed:
RUN pip install \
    deepspeed-kernels \
    scikit-learn \
    torch==${PYTORCH_VERSION}

##############################################################################
# DeepSpeed
##############################################################################
RUN <<EOR
  git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
  cd ${STAGE_DIR}/DeepSpeed
  git checkout .
  git checkout "tags/v${DEEPSPEED_VERSION}" -b "v${DEEPSPEED_VERSION}"
EOR

##############################################################################
# DeepSpeed build file
##############################################################################
RUN <<EOR
  cat << EOF > build_deepspeed.sh
#!/usr/bin/env bash
mkdir -p /deepspeed
cd ${STAGE_DIR}/DeepSpeed
DS_BUILD_OPS=1 python setup.py build_ext -j8 bdist_wheel
mv ${STAGE_DIR}/DeepSpeed/dist/*.whl /deepspeed/
EOF
EOR

RUN chmod +x /build_deepspeed.sh

ENTRYPOINT ["/build_deepspeed.sh"]