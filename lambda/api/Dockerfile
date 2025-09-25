FROM python:3.12-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.0 /lambda-adapter /opt/extensions/lambda-adapter

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    AWS_LAMBDA_EXEC_WRAPPER="/opt/bootstrap" \
    PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH="/health" \
    AWS_LWA_PORT=8080

WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-dev \
    libomp-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY app/ /app/
# スキーマディレクトリはコピーしない（DynamoDBから取得するため）

EXPOSE 8080

# 単一プロセスで実行（マルチワーカーは避ける）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
