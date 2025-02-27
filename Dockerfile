FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV HOST=0.0.0.0

RUN <<EOR
apt-get update
apt-get upgrade -y
apt-get install -y git build-essential portaudio19-dev \
  python3 python3-pip python-is-python3 gcc wget \
  ocl-icd-opencl-dev opencl-headers clinfo \
  libclblast-dev libopenblas-dev libaio-dev

# Need this 440MB dep on 22.04 otherwise TTS Analyze is very sad if we don't have 11.8 CUDA and lack the dep:
#   Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
#   https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2201088567
apt-get install -y libcudnn8

mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
apt-get clean && rm -rf /var/lib/apt/lists/*
EOR

WORKDIR /app
ENV CUDA_DOCKER_ARCH=all
COPY system/requirements/requirements_docker.txt system/requirements/requirements_docker.txt

COPY . .
RUN <<EOR
pip install --no-cache-dir --no-deps -r system/requirements/requirements_docker.txt
pip install --no-cache-dir deepspeed
EOR

EXPOSE 7851 7852
RUN chmod +x launch.sh
ENTRYPOINT ["sh", "-c", "./launch.sh"]
