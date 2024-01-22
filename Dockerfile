FROM python:3-slim-bullseye
ENV HOST 0.0.0.0
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements_other.txt
EXPOSE 7851
CMD ["python", "script.py"]
