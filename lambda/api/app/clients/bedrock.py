"""Bedrock API 呼び出し"""
import logging
from typing import Any, Callable

from botocore.exceptions import ClientError

from config import settings
from exceptions import ResponseParseError
from utils.bedrock import parse_converse_response
from .aws import create_bedrock_client

logger = logging.getLogger(__name__)

# モデルが応答を出し切ったことを示す停止理由。
RETRYABLE_STOP_REASONS = ("end_turn", "stop_sequence")

# 応答を読み取れなかったときの試行回数（初回 + 再試行）
PARSE_ATTEMPTS = 2

# temperature を受け付けないモデル ID（実行時に ValidationException で判明したもの）。
# OpenAI 系の reasoning モデルなど、temperature 指定を拒否するモデルがある。
_temperature_unsupported_models: set = set()


def _build_inference_config(model_id: str) -> dict:
    config = {"temperature": 0.2, "maxTokens": 40000}
    if model_id in _temperature_unsupported_models:
        del config["temperature"]
    return config


def _is_temperature_rejection(error: Exception) -> bool:
    """temperature 非対応モデルによる ValidationException か判定する"""
    if not isinstance(error, ClientError):
        return False
    if error.response.get("Error", {}).get("Code") != "ValidationException":
        return False
    return "temperature" in str(error).lower()


def call_bedrock(
    messages: list,
    system_prompts: list | None = None,
    model_id: str | None = None,
    model_region: str | None = None,
) -> dict:
    """Bedrock Converse API 呼び出し

    temperature を受け付けないモデルに対しては、初回の ValidationException を機に
    そのモデルを記憶し、temperature なしで再試行・以後の呼び出しでも送らない。
    """
    model_id = model_id or settings.MODEL_ID
    model_region = model_region or settings.MODEL_REGION
    bedrock = create_bedrock_client(model_region)

    try:
        response = bedrock.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=_build_inference_config(model_id),
        )
        return response
    except Exception as e:
        if _is_temperature_rejection(e) and model_id not in _temperature_unsupported_models:
            logger.warning(
                f"モデル {model_id} が temperature を受け付けないため、temperature なしで再試行します")
            _temperature_unsupported_models.add(model_id)
            return call_bedrock(messages, system_prompts, model_id, model_region)
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
    for attempt in range(PARSE_ATTEMPTS):
        response = call_bedrock(messages, system_prompts, **kwargs)
        stop_reason = response.get("stopReason") if isinstance(response, dict) else None
        try:
            return parse_fn(parse_converse_response(response))
        except ResponseParseError as e:
            if stop_reason not in RETRYABLE_STOP_REASONS:
                raise ResponseParseError(f"{e} (stopReason={stop_reason})") from e
            if attempt == PARSE_ATTEMPTS - 1:
                raise
            logger.warning(f"応答のパースに失敗したため再試行します: {e}")
