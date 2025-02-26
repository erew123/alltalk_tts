#!/usr/bin/env bash

declare -A ALLTALK_VARS
ALLTALK_VARS["CUDA_VERSION"]=12.6.0
ALLTALK_VARS["PYTHON_VERSION"]=3.11.11
ALLTALK_VARS["DEEPSPEED_VERSION"]=0.16.2
ALLTALK_VARS["ALLTALK_DIR"]=/opt/alltalk

# Export single variables (needed by Docker locally)
for key in "${!ALLTALK_VARS[@]}"
do
  export "${key}=${ALLTALK_VARS[${key}]}"
done

# Export the entire associative array (needed by Github action)
export ALLTALK_VARS