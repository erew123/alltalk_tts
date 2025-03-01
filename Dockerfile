ARG DOCKER_REPOSITORY
ARG DOCKER_TAG=latest
FROM ${DOCKER_REPOSITORY}alltalk_tts_environment:${DOCKER_TAG}

# Argument to choose the model: piper, vits, xtts
ARG TTS_MODEL="xtts"
ENV TTS_MODEL=$TTS_MODEL

ARG DEEPSPEED_VERSION=0.16.2
ENV DEEPSPEED_VERSION=$DEEPSPEED_VERSION

ARG ALLTALK_DIR=/opt/alltalk

ENV GRADIO_SERVER_NAME="0.0.0.0"

ENV ENABLE_MULTI_ENGINE_MANAGER=false
ENV WITH_UI=true

WORKDIR ${ALLTALK_DIR}

##############################################################################
# Install python dependencies (cannot use --no-deps because requirements are not complete)
##############################################################################
COPY system/config system/config
COPY system/requirements/requirements_standalone.txt system/requirements/requirements_standalone.txt
COPY system/requirements/requirements_parler.txt system/requirements/requirements_parler.txt
ENV PIP_CACHE_DIR=${ALLTALK_DIR}/pip_cache
RUN <<EOR
    conda activate alltalk

    mkdir ${ALLTALK_DIR}/pip_cache
    pip install --no-cache-dir --cache-dir=${ALLTALK_DIR}/pip_cache -r system/requirements/requirements_standalone.txt
    pip install --no-cache-dir --cache-dir=${ALLTALK_DIR}/pip_cache --upgrade gradio==4.32.2

    # By default, version 1.9.10 is used causing this warning on startup: 'FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated'
    pip install --no-cache-dir --cache-dir=${ALLTALK_DIR}/pip_cache local-attention==1.11.1

    # Parler:
    pip install --no-cache-dir --cache-dir=${ALLTALK_DIR}/pip_cache -r system/requirements/requirements_parler.txt

    conda clean --all --force-pkgs-dirs -y && pip cache purge
EOR

##############################################################################
# Install DeepSpeed
##############################################################################
RUN mkdir -p /tmp/deepspeed
COPY docker/deepspeed/build*/*.whl /tmp/deepspeed/
RUN <<EOR
    DEEPSPEED_WHEEL=$(realpath -q /tmp/deepspeed/*.whl)
    conda activate alltalk

    # Download DeepSpeed wheel if it was not built locally:
    if [ -z "${DEEPSPEED_WHEEL}" ] || [ ! -f $DEEPSPEED_WHEEL ] ; then
      echo "Downloading pre-built DeepSpeed wheel"
      CURL_ERROR=$( { curl --output-dir /tmp/deepspeed -fLO "https://github.com/erew123/alltalk_tts/releases/download/DeepSpeed-for-docker/deepspeed-0.16.2+b344c04d-cp311-cp311-linux_x86_64.whl" ; } 2>&1 )
      if [ $? -ne 0 ] ; then
        echo "Failed to download DeepSpeed: $CURL_ERROR"
        exit 1
      fi
      DEEPSPEED_WHEEL=$(realpath -q /tmp/deepspeed/*.whl)
    fi

    echo "Using precompiled DeepSpeed wheel at ${DEEPSPEED_WHEEL}"
    CFLAGS="-I$CONDA_PREFIX/include/" LDFLAGS="-L$CONDA_PREFIX/lib/" \
      pip install --no-cache-dir ${DEEPSPEED_WHEEL}

    if [ $? -ne 0 ] ; then
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

# Enabling or disabling UI:
jq ".launch_gradio = \$WITH_UI" docker_default_config.json > docker_default_config.json.tmp
mv docker_default_config.json.tmp docker_default_config.json

# Merging config from docker_confignew.json into confignew.json:
jq -s '.[0] * .[1] * .[2]' confignew.json docker_default_config.json docker_confignew.json  > confignew.json.tmp
mv confignew.json.tmp confignew.json

conda activate alltalk

if [ "\$ENABLE_MULTI_ENGINE_MANAGER" = true ] ; then
  echo "Starting alltalk using multi engine manager"
  # Merging config from docker_mem_config.json into mem_config.json:
  jq -s '.[0] * .[1] * .[2]' mem_config.json docker_default_mem_config.json docker_mem_config.json  > mem_config.json.tmp
  mv mem_config.json.tmp mem_config.json
  python tts_mem.py
else
  echo "Starting alltalk"
  python script.py
fi
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
