FROM ubuntu:22.04
ENV HOST 0.0.0.0
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential portaudio19-dev \
    python3 python3-pip gcc wget \
    && ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements_other.txt
EXPOSE 7851
CMD ["python", "script.py"]
