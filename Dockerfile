FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV HOST=0.0.0.0

RUN <<EOR
apt-get update
apt-get upgrade -y
apt-get install -y git build-essential portaudio19-dev \
  python3 python3-pip python-is-python3 gcc wget \
  ocl-icd-opencl-dev opencl-headers clinfo \
  libclblast-dev libopenblas-dev libaio-dev

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
