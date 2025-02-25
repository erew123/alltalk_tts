ARG GITHUB_REPOSITORY
ARG DOCKER_TAG=latest
FROM ${GITHUB_REPOSITORY}alltalk_tts_environment:${DOCKER_TAG}

ARG DEEPSPEED_VERSION=0.16.3
ENV DEEPSPEED_VERSION=$DEEPSPEED_VERSION

##############################################################################
# Directories:
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Install python dependencies
##############################################################################
ENV PIP_CACHE_DIR=/tmp/pip_cache
RUN <<EOR
  conda activate alltalk

  pip install \
      deepspeed-kernels
EOR


##############################################################################
# Clone DeepSpeed sources
##############################################################################
RUN <<EOR
  git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
  cd ${STAGE_DIR}/DeepSpeed
  git checkout .
  git checkout "tags/v${DEEPSPEED_VERSION}" -b "v${DEEPSPEED_VERSION}"

  # Patching some files due to bug https://github.com/microsoft/DeepSpeed/issues/6709
  START_LINE=`awk '/#ifndef BF16_AVAILABLE/{ print NR; exit }' csrc/transformer/inference/csrc/gelu.cu`
  TO_LINE=$((START_LINE + 2))
  sed -i "${START_LINE},${TO_LINE}d" ./csrc/transformer/inference/csrc/gelu.cu

  START_LINE=`awk '/#ifndef BF16_AVAILABLE/{ print NR; exit }' csrc/transformer/inference/csrc/transform.cu`
  TO_LINE=$((START_LINE + 2))
  sed -i "${START_LINE},${TO_LINE}d" ./csrc/transformer/inference/csrc/transform.cu
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