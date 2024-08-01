#!/bin/bash
cd "."
if [[ "/home/eleven/alltalkbeta/alltalk_tts" =~ " " ]]; then echo This script relies on Miniconda which can not be silently installed under a path with spaces. && exit; fi
# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null
# config
CONDA_ROOT_PREFIX="/home/eleven/alltalkbeta/alltalk_tts/alltalk_environment/conda"
INSTALL_ENV_DIR="/home/eleven/alltalkbeta/alltalk_tts/alltalk_environment/env"
# environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export CUDA_PATH="/home/eleven/alltalkbeta/alltalk_tts/alltalk_environment/env"
export CUDA_HOME="/home/eleven/alltalkbeta/alltalk_tts/alltalk_environment/env"
# activate env
bash --init-file <(echo "source \"/home/eleven/alltalkbeta/alltalk_tts/alltalk_environment/conda/etc/profile.d/conda.sh\" && conda activate \"/home/eleven/alltalkbeta/alltalk_tts/alltalk_environment/env\"")
