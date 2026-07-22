"""AWS クライアント一元管理パッケージ

後方互換: from clients import s3_client 等の既存 import をそのまま使用可能
"""
from .aws import (
    create_s3_client, create_bedrock_client, create_dynamodb_client,
    create_sagemaker_runtime_client, create_sagemaker_client,
    create_bedrock_agentcore_client, create_dynamodb_resource, create_sfn_client,
    s3_client, bedrock_client, dynamodb_client, dynamodb_resource,
    sagemaker_runtime_client, sagemaker_client, bedrock_agentcore_client, sfn_client,
    lambda_client, invoke_worker_async,
    get_inference_component_status, get_endpoint_status_direct, trigger_endpoint_wakeup,
)
from .agent import AgentClient
from .bedrock import call_bedrock, call_bedrock_with_retry

__all__ = [
    "create_s3_client", "create_bedrock_client", "create_dynamodb_client",
    "create_sagemaker_runtime_client", "create_sagemaker_client",
    "create_bedrock_agentcore_client", "create_dynamodb_resource", "create_sfn_client",
    "s3_client", "bedrock_client", "dynamodb_client", "dynamodb_resource",
    "sagemaker_runtime_client", "sagemaker_client", "bedrock_agentcore_client", "sfn_client",
    "lambda_client", "invoke_worker_async",
    "AgentClient",
    "call_bedrock", "call_bedrock_with_retry",
    "get_inference_component_status", "get_endpoint_status_direct", "trigger_endpoint_wakeup",
]
