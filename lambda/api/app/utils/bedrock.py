"""Bedrock レスポンスパース — 純粋関数のみ

API 呼び出し（call_bedrock, call_bedrock_with_retry）は clients/bedrock.py に移動済み。
"""
import logging
import json
import re

logger = logging.getLogger(__name__)


def parse_converse_response(response):
    """Converse API レスポンスからテキストを抽出する"""
    try:
        content = response["output"]["message"]["content"]
        if content and len(content) > 0:
            return content[0]["text"]
        else:
            logger.warning("レスポンスにテキストコンテンツが含まれていません")
            return ""
    except KeyError as e:
        logger.error(f"レスポンス解析エラー: {str(e)}")
        return ""


def extract_json_from_response(response_text):
    """レスポンステキストから JSON を抽出する"""
    try:
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            return json.loads(json_match.group())
        else:
            logger.warning("レスポンステキストにJSONが見つかりません")
            return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析エラー: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"JSON抽出エラー: {str(e)}")
        return {}
