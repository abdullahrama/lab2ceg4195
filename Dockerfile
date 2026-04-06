FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/
# Install CPU-only PyTorch first to keep the image small (~1 GB vs ~3 GB for CUDA).
# The container only runs inference; training is done locally with GPU.
RUN pip install --upgrade pip && \
    pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY lab2.py /app/
COPY tests/ /app/tests/

EXPOSE 5000

CMD ["python", "lab2.py"]
