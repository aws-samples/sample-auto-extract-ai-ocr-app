import os


class Settings:
    """アプリケーション設定"""

    # AWS設定
    BUCKET_NAME: str = os.getenv("BUCKET_NAME", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-1")

    # DynamoDB設定
    IMAGES_TABLE_NAME: str = os.getenv("IMAGES_TABLE_NAME", "")
    JOBS_TABLE_NAME: str = os.getenv("JOBS_TABLE_NAME", "")

    # 機能フラグ
    ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "true").lower() == "true"

    # Bedrock設定
    MODEL_ID: str = os.getenv(
        "MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    MODEL_REGION: str = os.getenv("MODEL_REGION", "us-east-1")

    # SageMaker設定
    SAGEMAKER_ENDPOINT_NAME: str = os.getenv(
        "SAGEMAKER_ENDPOINT_NAME", "")
    SAGEMAKER_INFERENCE_COMPONENT_NAME: str = os.getenv(
        "SAGEMAKER_INFERENCE_COMPONENT_NAME", "")

    # API設定
    API_BASE_URL: str = os.getenv("API_BASE_URL", "")


# グローバル設定インスタンス
settings = Settings()
