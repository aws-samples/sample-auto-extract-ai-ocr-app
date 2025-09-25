"""
Bedrock関連のユーティリティ関数
"""
import logging
import json
import time
import re
from typing import Dict, Any, List, Optional
from clients import create_bedrock_client
from config import settings

logger = logging.getLogger(__name__)


def call_bedrock(messages, system_prompts=None, model_id=None, model_region=None):
    """
    基本のBedrock Converse API呼び出し

    Args:
        messages (list): モデルに送信するメッセージのリスト
        system_prompts (list, optional): システムプロンプトのリスト
        model_id (str, optional): 使用するモデルID
        model_region (str, optional): モデルのリージョン

    Returns:
        dict: Bedrockからの生レスポンス
    """
    model_id = model_id or settings.MODEL_ID
    model_region = model_region or settings.MODEL_REGION

    logger.info(f"モデル {model_id} を使用して対話を実行します (リージョン: {model_region})")

    # 動的リージョン対応のため、専用クライアントを作成
    bedrock = create_bedrock_client(model_region)

    # 推論パラメータの設定
    inference_config = {
        "temperature": 0.2,
        "maxTokens": 40000
    }

    try:
        logger.info("Bedrock APIを呼び出し中")
        response = bedrock.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config
        )
        logger.info("Bedrock APIの呼び出しが成功しました")
        return response

    except Exception as e:
        logger.error(f"Bedrock API呼び出しエラー: {str(e)}")
        raise


def call_bedrock_with_retry(messages, system_prompts=None, max_retries=5):
    """
    リトライ付きBedrock呼び出し

    Args:
        messages (list): モデルに送信するメッセージのリスト
        system_prompts (list, optional): システムプロンプトのリスト
        max_retries (int): 最大リトライ回数

    Returns:
        dict: Bedrockからの生レスポンス
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Bedrock API呼び出し（試行回数: {attempt+1}/{max_retries}）")
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


def parse_converse_response(response):
    """
    Converse APIのレスポンスを解析してテキストを抽出する

    Args:
        response (dict): Converse APIからのレスポンス

    Returns:
        str: 抽出されたテキスト
    """
    try:
        content = response['output']['message']['content']
        if content and len(content) > 0:
            return content[0]['text']
        else:
            logger.warning("レスポンスにテキストコンテンツが含まれていません")
            return ""
    except KeyError as e:
        logger.error(f"レスポンス解析エラー: {str(e)}")
        return ""


def extract_json_from_response(response_text):
    """
    レスポンステキストからJSONを抽出する

    Args:
        response_text (str): レスポンステキスト

    Returns:
        dict: 抽出されたJSON、失敗時は空辞書
    """
    try:
        # JSONを含む部分を抽出
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            logger.warning("レスポンステキストにJSONが見つかりません")
            return {}

    except json.JSONDecodeError as e:
        logger.error(f"JSON解析エラー: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"JSON抽出エラー: {str(e)}")
        return {}


def parse_extraction_response(ai_response, field_names):
    """
    AI応答から抽出結果を解析してextracted_infoとmappingを分離

    Args:
        ai_response (str): AIからの応答テキスト
        field_names (list): 抽出対象のフィールド名リスト

    Returns:
        tuple: (extracted_info, mapping)
    """
    extracted_info = {}
    mapping = {}

    try:
        # Markdownのコードブロックを除去
        cleaned_text = ai_response.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]  # ```json を除去
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]   # ``` を除去
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]  # 末尾の ``` を除去
        cleaned_text = cleaned_text.strip()

        # JSONを含む部分を抽出
        json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
        if json_match:
            json_str = json_match.group(0)
            response_data = json.loads(json_str)

            # 統合形式のデータを解析
            if "extracted_data" in response_data and "indices" in response_data:
                extracted_info = response_data["extracted_data"]
                mapping = response_data["indices"]
                logger.info("統合形式でデータを解析しました")
            else:
                # 期待される形式でない場合はエラー
                logger.error(
                    f"期待される形式ではありません。キー: {list(response_data.keys())}")
                extracted_info = {
                    "error": "Invalid response format. Expected 'extracted_data' and 'indices' keys.",
                    "raw_response": ai_response
                }
                mapping = {field_name: [] for field_name in field_names}

            logger.info(
                f"LLMからマッピング情報を取得: {json.dumps(mapping, ensure_ascii=False, indent=0)}")
        else:
            extracted_info = {
                "error": "Failed to parse JSON from AI response",
                "raw_response": ai_response
            }
            mapping = {field_name: [] for field_name in field_names}
    except Exception as json_error:
        logger.error(f"Error parsing JSON: {str(json_error)}")
        extracted_info = {
            "error": f"JSON parsing error: {str(json_error)}",
            "raw_response": ai_response
        }
        mapping = {field_name: [] for field_name in field_names}

    return extracted_info, mapping
