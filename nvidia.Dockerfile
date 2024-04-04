FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV HOST 0.0.0.0
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential portaudio19-dev \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir /app/pip_cache
ENV PIP_CACHE_DIR=/app/pip_cache

WORKDIR /app
COPY . .
ENV CUDA_DOCKER_ARCH=all
RUN pip install --no-cache-dir --no-deps -r system/requirements/requirements_standalone.txt --cache-dir=/app/pip_cache && \
    pip install --no-cache-dir deepspeed --cache-dir=/app/pip_cache

EXPOSE 7851
RUN chmod +x launch.sh
ENTRYPOINT ["sh", "-c", "./launch.sh"]