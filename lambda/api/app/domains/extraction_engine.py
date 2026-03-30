"""情報抽出エンジン — 純粋なプロンプト構築 + レスポンスパース

Bedrock API 呼び出しは行わない。サービス層が担当する。
"""
from domains.prompts import (
    create_single_with_ocr_prompt, create_single_without_ocr_prompt,
    create_multi_with_ocr_prompt, create_multi_without_ocr_prompt
)
from utils.helpers import float_to_decimal
from domains.template import generate_unified_template
import logging
import json
import re
import base64

logger = logging.getLogger(__name__)


# ============================================================
# レスポンスパース（純粋関数）
# ============================================================

def parse_extraction_response(ai_response, field_names):
    """AI 応答から抽出結果を解析して extracted_info と mapping を分離"""
    extracted_info = {}
    mapping = {}

    try:
        cleaned_text = ai_response.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        json_match = re.search(r"\{[\s\S]*\}", cleaned_text)
        if json_match:
            json_str = json_match.group(0)
            response_data = json.loads(json_str)

            if "extracted_data" in response_data and "indices" in response_data:
                extracted_info = response_data["extracted_data"]
                mapping = response_data["indices"]
            else:
                logger.error(f"期待される形式ではありません。キー: {list(response_data.keys())}")
                extracted_info = {"error": "Invalid response format.", "raw_response": ai_response}
                mapping = {field_name: [] for field_name in field_names}
        else:
            extracted_info = {"error": "Failed to parse JSON from AI response", "raw_response": ai_response}
            mapping = {field_name: [] for field_name in field_names}
    except Exception as json_error:
        logger.error(f"Error parsing JSON: {str(json_error)}")
        extracted_info = {"error": f"JSON parsing error: {str(json_error)}", "raw_response": ai_response}
        mapping = {field_name: [] for field_name in field_names}

    return extracted_info, mapping


def finalize_extraction_result(extracted_info, mapping=None):
    """抽出結果を Decimal 変換して最終形式にする"""
    result = {"extracted_info": float_to_decimal(extracted_info)}
    if mapping is not None:
        result["mapping"] = float_to_decimal(mapping)
    return result


# ============================================================
# ヘルパー（純粋関数）
# ============================================================

def _generate_extraction_fields(fields, prefix=""):
    """フィールド定義から抽出対象の項目リストを生成"""
    result = []
    for field in fields:
        display_name = field["display_name"]
        field_type = field.get("type", "string")
        field_desc = f"{prefix} > {display_name} ({field_type}型)" if prefix else f"{display_name} ({field_type}型)"
        result.append(field_desc)
        if field_type == "map" and "fields" in field:
            result.extend(_generate_extraction_fields(field["fields"], display_name))
        elif field_type == "list" and "items" in field:
            items = field["items"]
            if items.get("type") == "map" and "fields" in items:
                result.extend(_generate_extraction_fields(items["fields"], f"{display_name} (各項目)"))
    return result


def _detect_image_format(image_bytes: bytes) -> str:
    """画像バイトからフォーマットを判定"""
    if image_bytes.startswith(b"\xff\xd8"):
        return "jpeg"
    elif image_bytes.startswith(b"\x89PNG"):
        return "png"
    return "jpeg"


# ============================================================
# メッセージ構築（純粋関数 — Bedrock 呼び出しなし）
# ============================================================

# 例示用データ（単一画像+OCR用）
_EXAMPLE_OCR = {
    "words": [
        {"id": 0, "content": "注文日：2023年5月1日", "points": [[50, 120], [250, 120], [250, 150], [50, 150]]},
        {"id": 1, "content": "委託業務内容：配送業務", "points": [[50, 180], [300, 180], [300, 210], [50, 210]]},
        {"id": 2, "content": "運行日：2023年5月15日", "points": [[50, 240], [250, 240], [250, 270], [50, 270]]},
        {"id": 3, "content": "A001", "points": [[50, 400], [100, 400], [100, 430], [50, 430]]},
        {"id": 4, "content": "東京", "points": [[150, 400], [200, 400], [200, 430], [150, 430]]},
        {"id": 5, "content": "大阪", "points": [[250, 400], [300, 400], [300, 430], [250, 430]]},
    ]
}

_EXAMPLE_OUTPUT = {
    "order_date": "2023年5月1日",
    "operation_info": {"contract_work": "配送業務", "operation_date": "2023年5月15日"},
    "shipment_details": [{"reception_number": "A001", "destination": "東京", "origin": "大阪", "vehicle_number": "", "fare": ""}],
    "indices": {
        "order_date": [0],
        "operation_info": {"contract_work": [1], "operation_date": [2]},
        "shipment_details": [{"reception_number": [3], "destination": [4], "origin": [5], "vehicle_number": [], "fare": []}],
    },
}


def build_single_image_with_ocr_request(
    image_data: bytes, content_type: str, ocr_result: dict,
    app_extraction_fields: dict, custom_prompt: str = ""
) -> tuple[list, list]:
    """単一画像+OCR のメッセージを構築する（Bedrock 呼び出しなし）

    Returns:
        (messages, system_prompts)
    """
    # 抽出対象の項目リストを生成
    extraction_fields = _generate_extraction_fields(app_extraction_fields["fields"])
    extraction_targets = "\n".join([f"{i+1}. {f}" for i, f in enumerate(extraction_fields)])
    # JSONテンプレートとindicesテンプレートを生成（templateモジュールを使用）
    unified_template = generate_unified_template(app_extraction_fields)

    prompt = create_single_with_ocr_prompt(
        extraction_targets, unified_template, _EXAMPLE_OCR, _EXAMPLE_OUTPUT,
        ocr_result, custom_prompt
    )

    system_prompts = [{"text": "あなたはOCR結果から情報を抽出するアシスタントです。指定されたフィールドに対応する情報を抽出し、JSONフォーマットで返してください。"}]

    image_format = content_type.split("/")[1] if content_type and "/" in content_type else "jpeg"

    if image_data:
        messages = [{"role": "user", "content": [
            {"image": {"format": image_format, "source": {"bytes": image_data}}},
            {"text": prompt},
        ]}]
    else:
        messages = [{"role": "user", "content": [{"text": prompt}]}]

    return messages, system_prompts


def build_multi_images_with_ocr_request(
    page_images: list, content_type: str, ocr_results: list,
    app_extraction_fields: dict, custom_prompt: str = ""
) -> tuple[list, list]:
    """複数画像+OCR のメッセージを構築する

    Returns:
        (messages, system_prompts)
    """
    if not page_images:
        raise ValueError("画像データを取得できませんでした")
    if not ocr_results:
        raise ValueError("OCR結果が見つかりません")

    instructions = "以下のスキーマに従って、文書から情報を抽出してください。"
    prompt = create_multi_with_ocr_prompt(ocr_results, app_extraction_fields, instructions, custom_prompt)

    system_prompts = [{"text": "あなたは複数ページの文書から情報を抽出するアシスタントです。指定されたフィールドに対応する情報を抽出し、純粋なJSONオブジェクトのみを返してください。説明文、コメント、マークダウン記法は一切使用しないでください。"}]

    image_format = content_type.split("/")[1] if content_type and "/" in content_type else "jpeg"
    content = [{"text": prompt}]
    for image_bytes in page_images:
        content.append({"image": {"format": image_format, "source": {"bytes": image_bytes}}})

    messages = [{"role": "user", "content": content}]
    return messages, system_prompts


def build_multi_images_without_ocr_request(
    images_data: list, app_extraction_fields: dict,
    field_names: list, custom_prompt: str = ""
) -> tuple[list, list]:
    """OCRなし複数画像のメッセージを構築する

    Args:
        images_data: [{"bytes": bytes, "content_type": str}, ...]

    Returns:
        (messages, system_prompts)
    """
    if not images_data:
        raise ValueError("画像データが見つかりません")

    vision_prompt = create_multi_without_ocr_prompt(
        app_extraction_fields.get("fields", []), field_names, custom_prompt
    )

    system_prompts = [{"text": "あなたは複数の画像から情報を抽出するアシスタントです。画像を直接解析して、指定されたフィールドに対応する情報を抽出し、JSONフォーマットで返してください。"}]

    content = []
    for img_data in images_data:
        ct = img_data["content_type"]
        fmt = ct.split("/")[1] if ct and "/" in ct else "jpeg"
        content.append({"image": {"format": fmt, "source": {"bytes": img_data["bytes"]}}})
    content.append({"text": vision_prompt})

    messages = [{"role": "user", "content": content}]
    return messages, system_prompts


def build_single_image_without_ocr_request(
    image_bytes: bytes, app_extraction_fields: dict,
    field_names: list, custom_prompt: str = ""
) -> tuple[list, list]:
    """OCRなし単一画像のメッセージを構築する

    Returns:
        (messages, system_prompts)
    """
    vision_prompt = create_single_without_ocr_prompt(
        app_extraction_fields.get("fields", []), field_names, custom_prompt
    )

    image_format = _detect_image_format(image_bytes)

    messages = [{"role": "user", "content": [
        {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
        {"text": vision_prompt},
    ]}]

    system_prompts = [{"text": "あなたは画像から情報を抽出するアシスタントです。画像を直接解析して、指定されたフィールドに対応する情報を抽出し、JSONフォーマットで返してください。"}]

    return messages, system_prompts
