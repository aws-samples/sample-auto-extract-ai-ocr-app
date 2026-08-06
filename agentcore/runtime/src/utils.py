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
