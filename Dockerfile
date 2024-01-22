ARG image_name=python:3.11-slim
FROM ${image_name}
WORKDIR /app
COPY . .
# Argument for specifying the GPU type, defaulting to "other"
ARG gpu_type=other
RUN pip install --no-cache-dir -r requirements_${gpu_type}.txt
EXPOSE 7851
CMD ["python", "script.py"]
