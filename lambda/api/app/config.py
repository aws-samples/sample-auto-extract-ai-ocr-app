import os


class Settings:
    """アプリケーション設定"""

    # AWS設定
    BUCKET_NAME: str = os.getenv("BUCKET_NAME", "")
    SYNC_BUCKET_NAME: str = os.getenv("SYNC_BUCKET_NAME", "")
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
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "paddle")

    # API設定
    API_BASE_URL: str = os.getenv("API_BASE_URL", "")
    
    # Agent設定
    AGENT_RUNTIME_ARN: str = os.getenv("AGENT_RUNTIME_ARN", "")
    AGENT_KICK_FUNCTION_NAME: str = os.getenv("AGENT_KICK_FUNCTION_NAME", "")

    # Schema Generation Worker Lambda
    SCHEMA_GENERATE_FUNCTION_NAME: str = os.getenv("SCHEMA_GENERATE_FUNCTION_NAME", "")

    # PDF Convert Worker Lambda
    PDF_CONVERT_FUNCTION_NAME: str = os.getenv("PDF_CONVERT_FUNCTION_NAME", "")

    # S3 Sync Import Worker Lambda
    S3_SYNC_IMPORT_FUNCTION_NAME: str = os.getenv("S3_SYNC_IMPORT_FUNCTION_NAME", "")

    # Step Functions設定
    STATE_MACHINE_ARN: str = os.getenv("STATE_MACHINE_ARN", "")


# グローバル設定インスタンス
settings = Settings()
