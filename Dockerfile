FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements_nvidia.txt && \
    pip install --no-cache-dir -r requirements_finetune.txt && \
    pip install --no-cache-dir -r requirements_other.txt
EXPOSE 7851
CMD ["python", "script.py"]
