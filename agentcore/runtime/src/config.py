"""Configuration for agent runtime."""

import logging
import os
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 20


def get_aws_credentials() -> dict[str, str]:
    """Get AWS credentials from environment"""
    credentials = {}
    
    if "AWS_ACCESS_KEY_ID" in os.environ:
        credentials["AWS_ACCESS_KEY_ID"] = os.environ["AWS_ACCESS_KEY_ID"]
    if "AWS_SECRET_ACCESS_KEY" in os.environ:
        credentials["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    if "AWS_SESSION_TOKEN" in os.environ:
        credentials["AWS_SESSION_TOKEN"] = os.environ["AWS_SESSION_TOKEN"]
    
    credentials["AWS_REGION"] = os.environ.get("AWS_REGION", "us-east-1")
    
    return credentials


def get_system_prompt(user_system_prompt: str = None) -> str:
    """Combine user system prompt with fixed instructions"""
    fixed_prompt = """あなたはOCR抽出結果を検証し、誤りを修正するアシスタントです。

## 指示
- 画像（原本）と抽出結果を照合し、差異がないか確認してください
- 利用可能なツールを使って、抽出値の正確性を検証してください
- 数値の計算や合計に誤りがないか、ツールで検算してください
- 外部データとの照合が可能な場合は、ツールで正確な情報を取得して比較してください
- 修正提案は必ずツールによる検証結果に基づいてください

## 制約
- 与えられたツールのみを使用してください
- ツールで検証できない項目について推測で修正提案しないでください

## 出力形式
修正が必要な場合のみ、以下のJSON形式で出力してください：
{
  "suggestions": [
    {
      "field": "フィールド名（ネストはドット区切り。例: client_info.address）",
      "original_value": "元の値",
      "suggested_value": "修正後の値",
      "reason": "修正理由（具体的に何が間違っていたか）",
      "confidence": "high" | "medium" | "low",
      "tool_used": "使用したツール名（必須）"
    }
  ]
}

修正が不要な場合は空の配列を返してください：
{
  "suggestions": []
}
"""

    if user_system_prompt:
        return f"{user_system_prompt}\n\n{fixed_prompt}"
    else:
        return fixed_prompt


def extract_model_info(model_info: Any) -> tuple[str, str]:
    """Extract model ID and region from model info"""
    aws_creds = get_aws_credentials()
    
    if isinstance(model_info, str):
        model_id = model_info
        region = aws_creds.get("AWS_REGION", "us-east-1")
    else:
        model_id = model_info.get("modelId", "us.anthropic.claude-sonnet-4-20250514-v1:0")
        region = model_info.get("region", aws_creds.get("AWS_REGION", "us-east-1"))
    
    return model_id, region


def get_max_iterations() -> int:
    """Get maximum iterations from environment"""
    try:
        return int(os.environ.get("MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS))
    except ValueError:
        logger.warning(f"Invalid MAX_ITERATIONS value. Defaulting to {DEFAULT_MAX_ITERATIONS}.")
        return DEFAULT_MAX_ITERATIONS
