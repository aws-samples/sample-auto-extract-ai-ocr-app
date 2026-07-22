"""AWS SDK クライアント生成 + グローバルインスタンス"""
import json
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


def create_sfn_client():
    """Step Functions クライアントを作成"""
    return boto3.client("stepfunctions", region_name=settings.AWS_REGION)


def create_lambda_client():
    """Lambda クライアントを作成（Worker への async invoke 用）"""
    return boto3.client("lambda", region_name=settings.AWS_REGION)


# グローバルインスタンス
s3_client = create_s3_client()
bedrock_client = create_bedrock_client()
dynamodb_client = create_dynamodb_client()
dynamodb_resource = create_dynamodb_resource()
sagemaker_runtime_client = create_sagemaker_runtime_client()
sagemaker_client = create_sagemaker_client()
bedrock_agentcore_client = create_bedrock_agentcore_client()
sfn_client = create_sfn_client()
lambda_client = create_lambda_client()


def invoke_worker_async(function_name: str, payload: dict) -> None:
    """Worker Lambda を非同期（InvocationType="Event"）で起動する。

    単発 fire-and-forget の Worker 起動を1箇所に集約する。
    """
    lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps(payload),
    )


def get_inference_component_status(component_name: str) -> dict:
    """推論コンポーネントの状態を取得"""
    response = sagemaker_client.describe_inference_component(
        InferenceComponentName=component_name
    )
    copy_count = response['RuntimeConfig']['CurrentCopyCount']
    return {
        'ready': copy_count > 0,
        'copy_count': copy_count,
        'status': 'ready' if copy_count > 0 else 'cold'
    }


def get_endpoint_status_direct(endpoint_name: str) -> dict:
    """InferenceComponent を使わないエンドポイントの状態取得（Marketplace モデル用）"""
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    return {
        'ready': status == 'InService',
        'status': status.lower(),
    }


def trigger_endpoint_wakeup(endpoint_name: str, component_name: str):
    """エンドポイントのスケールアウトをトリガー（ダミーリクエスト送信）"""
    try:
        sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            InferenceComponentName=component_name,
            Body='{"dummy": true}',
            ContentType='application/json'
        )
    except Exception:
        # NoCapacityエラーが期待される（これがスケールアウトをトリガー）
        pass
