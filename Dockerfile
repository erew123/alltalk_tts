FROM continuumio/miniconda3:24.7.1-0

# Argument to choose the model: piper, vits, xtts
ARG TTS_MODEL="xtts"
ENV TTS_MODEL=$TTS_MODEL

ARG CUDA_VERSION="12.1.1"
ENV CUDA_VERSION=$CUDA_VERSION

ARG PYTHON_VERSION=3.11.9
ENV PYTHON_VERSION=$PYTHON_VERSION

ARG PYTORCH_VERSION=2.2.1
ENV PYTORCH_VERSION=$PYTORCH_VERSION

SHELL ["/bin/bash", "-l", "-c"]
ENV SHELL=/bin/bash
ENV HOST=0.0.0.0
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV NVIDIA_VISIBLE_DEVICES=all

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

WORKDIR /alltalk

# Create a conda environment and install dependencies:
ARG INSTALL_ENV_DIR=/alltalk/alltalk_environment/env
ENV CONDA_AUTO_UPDATE_CONDA="false"
RUN <<EOR
    CUDA_SHORT_VERSION=${CUDA_VERSION%.*}

    conda create -y -n "alltalk" -c conda-forge python=${PYTHON_VERSION}
    conda activate alltalk
    RESULT=$( { conda install -y \
      gcc_linux-64 \
      gxx_linux-64 \
      pytorch=${PYTORCH_VERSION} \
      pytorch-cuda=${CUDA_SHORT_VERSION} \
      torchvision \
      torchaudio \
      libaio \
      nvidia/label/cuda-${CUDA_SHORT_VERSION}.0::cuda-toolkit \
      faiss-gpu=1.9.0 \
      conda-forge::ffmpeg=7.1.0 \
      conda-forge::portaudio=19.7.0 \
      -c pytorch \
      -c anaconda \
      -c nvidia ; } 2>&1 )

    if echo $RESULT | grep -izq error ; then
      echo "Failed to install conda dependencies 2: $RESULT"
      exit 1
    fi
    conda clean -a && pip cache purge
EOR

# Install python dependencies (cannot use --no-deps because requirements are not complete)
COPY system/config system/config
COPY system/requirements/requirements_standalone.txt system/requirements/requirements_standalone.txt
COPY system/requirements/requirements_parler.txt system/requirements/requirements_parler.txt
ENV PIP_CACHE_DIR=/alltalk/pip_cache
RUN <<EOR
    conda activate alltalk

    mkdir /alltalk/pip_cache
    pip install --no-cache-dir --cache-dir=/alltalk/pip_cache -r system/requirements/requirements_standalone.txt
    pip install --no-cache-dir --cache-dir=/alltalk/pip_cache --upgrade gradio==4.32.2
    # Parler:
    pip install --no-cache-dir --cache-dir=/alltalk/pip_cache -r system/requirements/requirements_parler.txt

    conda clean --all --force-pkgs-dirs -y && pip cache purge
EOR

# Deepspeed:
RUN mkdir -p /tmp/deepseped
COPY deepspeed/build/*.whl /tmp/deepspeed/
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

### Deepspeed requires cutlass:
###RUN git clone --depth 1 --branch "v3.5.1" https://github.com/NVIDIA/cutlass /alltalk/cutlass
###ENV CUTLASS_PATH=/alltalk/cutlass

# Writing scripts to start alltalk:
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

# Create script to execute firstrun.py:
RUN echo $'#!/usr/bin/env bash \n\
source ~/.bashrc \n\
conda activate alltalk \n\
python ./system/config/firstrun.py $@' > ./start_firstrun.sh

RUN chmod +x start_firstrun.sh
RUN ./start_firstrun.sh --tts_model $TTS_MODEL

RUN mkdir -p /alltalk/outputs
RUN mkdir -p /root/.triton/autotune

# Enabling deepspeed for all models:
RUN find . -name model_settings.json -exec sed -i -e 's/"deepspeed_enabled": false/"deepspeed_enabled": true/g' {} \;

# Downloading all RVC models:
RUN <<EOR
  jq -r '.[]' system/tts_engines/rvc_files.json > /tmp/rvc_files.txt
  xargs -n 1 curl --create-dirs --output-dir models/rvc_base -LO < /tmp/rvc_files.txt
  rm -f /tmp/rvc_files.txt
EOR

## Start alltalk:
ENTRYPOINT ["sh", "-c", "./start_alltalk.sh"]
