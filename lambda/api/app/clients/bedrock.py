"""Bedrock API 呼び出し"""
import logging
from typing import Any, Callable

from config import settings
from exceptions import ResponseParseError
from utils.bedrock import parse_converse_response
from .aws import create_bedrock_client

logger = logging.getLogger(__name__)

# モデルが応答を出し切ったことを示す停止理由。
RETRYABLE_STOP_REASONS = ("end_turn", "stop_sequence")

PARSE_FAILURE_ATTEMPTS = 2


def call_bedrock(
    messages: list,
    system_prompts: list | None = None,
    model_id: str | None = None,
    model_region: str | None = None,
) -> dict:
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


def call_bedrock_and_parse(
    messages: list,
    system_prompts: list | None,
    parse_fn: Callable[[str], Any],
    **kwargs,
) -> Any:
    """Bedrock を呼び、応答を parse_fn で読み取る。読み取れなければ再試行する。

    モデルが応答を出し切った（RETRYABLE_STOP_REASONS）のに読み取れない場合は、
    もう一度生成させれば通る見込みがあるため再試行する。それ以外の停止理由は
    打ち切られた応答なので、再実行しても同じく打ち切られる見込みが高く即座に失敗させる。

    API エラーの再試行は boto3 クライアントの retries 設定（clients/aws.py）に任せる。

    Args:
        messages: Converse API の messages
        system_prompts: Converse API の system
        parse_fn: 応答テキストを受け取り、読み取れない場合は
            ResponseParseError を投げる関数
        **kwargs: call_bedrock に渡す追加引数（model_id, model_region）

    Raises:
        ResponseParseError: 再試行しても応答を読み取れなかった場合
    """
    for attempt in range(PARSE_FAILURE_ATTEMPTS):
        response = call_bedrock(messages, system_prompts, **kwargs)
        stop_reason = response.get("stopReason") if isinstance(response, dict) else None
        try:
            return parse_fn(parse_converse_response(response))
        except ResponseParseError as e:
            if stop_reason not in RETRYABLE_STOP_REASONS:
                raise ResponseParseError(f"{e} (stopReason={stop_reason})") from e
            if attempt == PARSE_FAILURE_ATTEMPTS - 1:
                raise
            logger.warning(f"応答のパースに失敗したため再試行します: {e}")
