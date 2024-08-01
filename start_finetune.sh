#!/bin/bash
export TRAINER_TELEMETRY=0
source "/home/eleven/alltalkbeta/alltalk_tts/alltalk_environment/conda/etc/profile.d/conda.sh"
conda activate "/home/eleven/alltalkbeta/alltalk_tts/alltalk_environment/env"
python finetune.py
