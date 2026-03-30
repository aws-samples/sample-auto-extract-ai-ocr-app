"""AWS SDK クライアント生成 + グローバルインスタンス"""
import boto3
from botocore.config import Config
from config import settings


def create_s3_client():
    """リージョン固定・バケット仮想ホスト名対応のS3クライアントを作成"""
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},  # バケット仮想ホスト名を使用
        ),
    )


def create_bedrock_client(region_name=None):
    """Bedrock Runtime クライアントを作成

    Args:
        region_name: リージョン名。未指定時は settings.MODEL_REGION を使用
    """
    return boto3.client(
        "bedrock-runtime",
        region_name=region_name or settings.MODEL_REGION,
        config=Config(
            read_timeout=900,  # 15分のタイムアウト
            retries={"max_attempts": 3},
        ),
    )


def create_dynamodb_client():
    """DynamoDB クライアントを作成"""
    return boto3.client("dynamodb", region_name=settings.AWS_REGION)


def create_sagemaker_runtime_client():
    """SageMaker Runtime クライアントを作成"""
    return boto3.client("runtime.sagemaker", region_name=settings.AWS_REGION)


def create_sagemaker_client():
    """SageMaker クライアントを作成（エンドポイント管理用）"""
    return boto3.client("sagemaker", region_name=settings.AWS_REGION)


def create_bedrock_agentcore_client():
    """Bedrock AgentCore クライアントを作成"""
    return boto3.client(
        "bedrock-agentcore",
        region_name=settings.AWS_REGION,
        config=Config(
            read_timeout=300,  # 5分のタイムアウト
            retries={"max_attempts": 3},
        ),
    )


def create_dynamodb_resource():
    return boto3.resource("dynamodb", region_name=settings.AWS_REGION)


# グローバルインスタンス
s3_client = create_s3_client()
bedrock_client = create_bedrock_client()
dynamodb_client = create_dynamodb_client()
dynamodb_resource = create_dynamodb_resource()
sagemaker_runtime_client = create_sagemaker_runtime_client()
sagemaker_client = create_sagemaker_client()
bedrock_agentcore_client = create_bedrock_agentcore_client()
