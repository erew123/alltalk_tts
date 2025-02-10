FROM continuumio/miniconda3:24.7.1-0

# Argument to choose the model: piper, vits, xtts
ARG TTS_MODEL="xtts"
ENV TTS_MODEL=$TTS_MODEL

ARG ALLTALK_DIR=/opt/alltalk

SHELL ["/bin/bash", "-l", "-c"]
ENV SHELL=/bin/bash
ENV HOST=0.0.0.0
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CONDA_AUTO_UPDATE_CONDA="false"

ENV GRADIO_SERVER_NAME="0.0.0.0"

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

WORKDIR ${ALLTALK_DIR}

##############################################################################
# Create a conda environment and install dependencies:
##############################################################################
COPY docker/conda/build/environment-*.yml environment.yml
RUN <<EOR
    RESULT=$( { conda env create -f environment.yml ; } 2>&1 )

    if [ ! -f /environment-cu-${CUDA_VERSION}-cp-${PYTHON_VERSION}.yml] ; then
      echo "Failed to install conda dependencies: $RESULT"
      exit 1
    fi

    conda clean -a && pip cache purge
EOR

##############################################################################
# Install python dependencies (cannot use --no-deps because requirements are not complete)
##############################################################################
COPY system/config system/config
COPY system/requirements/requirements_standalone.txt system/requirements/requirements_standalone.txt
COPY system/requirements/requirements_parler.txt system/requirements/requirements_parler.txt
ENV PIP_CACHE_DIR=${ALLTALK_DIR}/pip_cache
RUN <<EOR
    conda activate alltalk

    mkdir ${ALLTALK_DIR}k/pip_cache
    pip install --no-cache-dir --cache-dir=${ALLTALK_DIR}/pip_cache -r system/requirements/requirements_standalone.txt
    pip install --no-cache-dir --cache-dir=${ALLTALK_DIR}/pip_cache --upgrade gradio==4.32.2
    # Parler:
    pip install --no-cache-dir --cache-dir=${ALLTALK_DIR}/pip_cache -r system/requirements/requirements_parler.txt

    conda clean --all --force-pkgs-dirs -y && pip cache purge
EOR

##############################################################################
# Install DeepSpeed
##############################################################################
RUN mkdir -p /tmp/deepseped
COPY docker/deepspeed/build/*.whl /tmp/deepspeed/
RUN <<EOR
    DEEPSPEED_WHEEL=$(realpath /tmp/deepspeed/*.whl)
    conda activate alltalk

    RESULT=$( { CFLAGS="-I$CONDA_PREFIX/include/" LDFLAGS="-L$CONDA_PREFIX/lib/" \
      pip install --no-cache-dir ${DEEPSPEED_WHEEL} ; } 2>&1 )

    if echo $RESULT | grep -izq error ; then
      echo "Failed to install pip dependencies: $RESULT"
      exit 1
    fi

    rm ${DEEPSPEED_WHEEL}
    conda clean --all --force-pkgs-dirs -y && pip cache purge
EOR

##############################################################################
# Writing scripts to start alltalk:
##############################################################################
RUN <<EOR
    cat << EOF > start_alltalk.sh
#!/usr/bin/env bash
source ~/.bashrc

# Merging config from docker_confignew.json into confignew.json:
jq -s '.[0] * .[1] * .[2]' confignew.json docker_default_config.json docker_confignew.json  > confignew.json.tmp
mv confignew.json.tmp confignew.json

conda activate alltalk
python script.py
EOF
    cat << EOF > start_finetune.sh
#!/usr/bin/env bash
source ~/.bashrc
export TRAINER_TELEMETRY=0
conda activate alltalk
python finetune.py
EOF
    cat << EOF > start_diagnostics.sh
#!/usr/bin/env bash
source ~/.bashrc
conda activate alltalk
python diagnostics.py
EOF
    chmod +x start_alltalk.sh
    chmod +x start_environment.sh
    chmod +x start_finetune.sh
    chmod +x start_diagnostics.sh
EOR

COPY . .

##############################################################################
# Create script to execute firstrun.py and run it:
##############################################################################
RUN echo $'#!/usr/bin/env bash \n\
  source ~/.bashrc \n\
  conda activate alltalk \n\
  python ./system/config/firstrun.py $@' > ./start_firstrun.sh

RUN chmod +x start_firstrun.sh
RUN ./start_firstrun.sh --tts_model $TTS_MODEL

RUN mkdir -p ${ALLTALK_DIR}/outputs
RUN mkdir -p /root/.triton/autotune

##############################################################################
# Enable deepspeed for all models:
##############################################################################
RUN find . -name model_settings.json -exec sed -i -e 's/"deepspeed_enabled": false/"deepspeed_enabled": true/g' {} \;

##############################################################################
# Download all RVC models:
##############################################################################
RUN <<EOR
  jq -r '.[]' system/tts_engines/rvc_files.json > /tmp/rvc_files.txt
  xargs -n 1 curl --create-dirs --output-dir models/rvc_base -LO < /tmp/rvc_files.txt
  rm -f /tmp/rvc_files.txt
EOR

##############################################################################
# Start alltalk:
##############################################################################
ENTRYPOINT ["sh", "-c", "./start_alltalk.sh"]
