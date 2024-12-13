FROM continuumio/miniconda3:24.7.1-0

# Argument to choose the model: piper, vits, xtts
ARG TTS_MODEL="xtts"
ENV TTS_MODEL=$TTS_MODEL

SHELL ["/bin/bash", "-l", "-c"]
ENV SHELL=/bin/bash
ENV HOST=0.0.0.0
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all
ENV GRADIO_SERVER_NAME="0.0.0.0"

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
    conda create -y -n "alltalk" -c conda-forge python=3.11.9
    conda activate alltalk
    conda install -y \
      gcc_linux-64 \
      gxx_linux-64 \
      pytorch=2.2.1 \
      torchvision \
      torchaudio \
      pytorch-cuda=12.1 \
      libaio \
      nvidia/label/cuda-12.1.0::cuda-toolkit=12.1 \
      faiss-gpu=1.9.0 \
      conda-forge::ffmpeg=7.1.0 \
      conda-forge::portaudio=19.7.0 \
      -c pytorch \
      -c anaconda \
      -c nvidia | grep -zq PackagesNotFoundError && exit 1
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

    # Deepspeed:
    curl -LO https://github.com/erew123/alltalk_tts/releases/download/DeepSpeed-14.0/deepspeed-0.14.2+cu121torch2.2-cp311-cp311-manylinux_2_24_x86_64.whl
    CFLAGS="-I$CONDA_PREFIX/include/" LDFLAGS="-L$CONDA_PREFIX/lib/" \
      pip install --no-cache-dir --cache-dir=/alltalk/pip_cache deepspeed-0.14.2+cu121torch2.2-cp311-cp311-manylinux_2_24_x86_64.whl
    rm -f deepspeed-0.14.2+cu121torch2.2-cp311-cp311-manylinux_2_24_x86_64.whl

    # Parler:
    pip install --no-cache-dir --no-deps --cache-dir=/alltalk/pip_cache -r system/requirements/requirements_parler.txt

    conda clean --all --force-pkgs-dirs -y && pip cache purge
EOR

# Deepspeed requires cutlass:
RUN git clone --depth 1 --branch "v3.5.1" https://github.com/NVIDIA/cutlass /alltalk/cutlass
ENV CUTLASS_PATH=/alltalk/cutlass

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
