"""Utility functions for agent runtime."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def process_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process messages for Strands Agent"""
    return messages


def process_prompt(prompt: str | list[dict[str, Any]]) -> str:
    """Process prompt for Strands Agent"""
    if isinstance(prompt, str):
        return prompt
    elif isinstance(prompt, list):
        text_parts = []
        for item in prompt:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return "\n".join(text_parts)
    return ""


def _contains_bytes(obj: Any) -> bool:
    if isinstance(obj, (bytes, bytearray)):
        return True
    if isinstance(obj, dict):
        return any(_contains_bytes(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_bytes(v) for v in obj)
    return False


def sanitize_message(message: Any) -> Any:
    """メッセージを JSON 直列化可能な形に整える。

    reasoning モデルは content に reasoningContent ブロックを含み、その
    redactedContent（暗号化された reasoning）は bytes のため JSON 化できない。
    呼び出し側（Lambda）は text ブロックしか使わないので、bytes を含むブロックを
    content から除いて返す。
    """
    if not isinstance(message, dict):
        return message
    content = message.get("content")
    if not isinstance(content, list):
        return message

    sanitized_content = [
        block for block in content
        if not _contains_bytes(block)
    ]
    if len(sanitized_content) != len(content):
        logger.info(
            f"JSON 化できないブロックを応答から除外しました: {len(content) - len(sanitized_content)} 件")
    return {**message, "content": sanitized_content}
