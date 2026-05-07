"""Bedrock API 呼び出し + リトライ"""
import logging
import time
from config import settings
from .aws import create_bedrock_client

logger = logging.getLogger(__name__)


def call_bedrock(messages, system_prompts=None, model_id=None, model_region=None):
    """Bedrock Converse API 呼び出し"""
    model_id = model_id or settings.MODEL_ID
    model_region = model_region or settings.MODEL_REGION
    bedrock = create_bedrock_client(model_region)
    inference_config = {"temperature": 0.2, "maxTokens": 40000}

    try:
        response = bedrock.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
        )
        return response
    except Exception as e:
        logger.error(f"Bedrock API呼び出しエラー: {str(e)}")
        raise


def call_bedrock_with_retry(messages, system_prompts=None, max_retries=5):
    """リトライ付き Bedrock 呼び出し（指数バックオフ）"""
    for attempt in range(max_retries):
        try:
            return call_bedrock(messages, system_prompts)
        except Exception as e:
            logger.error(f"Bedrock API呼び出しエラー: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数バックオフ
                logger.info(f"{wait_time}秒待機してリトライします...")
                time.sleep(wait_time)
            else:
                logger.error(f"最大試行回数 {max_retries} 回で失敗しました")
                raise
    return None
