FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create Python command symbolic links
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PaddlePaddle GPU version
RUN pip install --no-cache-dir paddlepaddle-gpu==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install Python packages
COPY requirements.txt /opt/ml/code/
RUN pip install --no-cache-dir -r requirements.txt

# Install PaddleOCR
RUN pip install --no-cache-dir paddleocr>=3.0.0

# Copy inference code
COPY inference.py /opt/ml/code/

# Expose SageMaker port
EXPOSE 8080

# Start application
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]
