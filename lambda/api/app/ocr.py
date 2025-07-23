import cv2
import numpy as np
import json
import logging
import os
import base64
import boto3
import uuid
import re
import time
from decimal import Decimal

# DynamoDB Decimal型をJSON serializable に変換するヘルパー関数
def decimal_to_float(obj):
    """Decimal型をfloat型に変換してJSON serializable にする"""
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj
from botocore.exceptions import ClientError

from app_schema import get_extraction_fields_for_app, get_field_names_for_app, DEFAULT_APP
from database import get_image, update_extracted_info, update_image_status

# ログ設定
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')
runtime_sm_client = boto3.client('runtime.sagemaker')

# SageMakerエンドポイント名とInferenceComponent名（環境変数から取得）
SAGEMAKER_ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME", "yomitoku-endpoint")
SAGEMAKER_INFERENCE_COMPONENT_NAME = os.getenv(
    "SAGEMAKER_INFERENCE_COMPONENT_NAME", "yomitoku-inference-component")

# モデルIDとリージョンを環境変数から取得
MODEL_ID = os.environ.get(
    'MODEL_ID', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
MODEL_REGION = os.environ.get('MODEL_REGION', 'us-east-1')

# OCR有効フラグを取得
ENABLE_OCR = os.environ.get('ENABLE_OCR', 'true').lower() == 'true'


def perform_ocr(image_data):
    """画像データに対してOCR処理を実行し、結果を返す（SageMakerエンドポイント使用）"""
    if not ENABLE_OCR:
        raise ValueError("OCR is disabled in this deployment")

    if not SAGEMAKER_ENDPOINT_NAME:
        raise ValueError("SageMaker endpoint not configured")

    try:
        logger.info(
            f"SageMakerエンドポイント {SAGEMAKER_ENDPOINT_NAME} を使用してOCR処理を実行中")

        # 画像をBase64エンコード
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # SageMakerエンドポイントへのリクエストデータを作成
        request_body = {
            "image": image_base64
        }

        # SageMakerエンドポイントを呼び出し
        try:
            # 推論コンポーネントを直接指定してエンドポイントを呼び出し
            response = runtime_sm_client.invoke_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps(request_body),
                InferenceComponentName=SAGEMAKER_INFERENCE_COMPONENT_NAME  # 直接パラメータとして指定
            )

            # レスポンスを解析
            response_body = json.loads(response['Body'].read().decode('utf-8'))

            # エラーチェック
            if 'error' in response_body:
                logger.error(
                    f"SageMakerエンドポイントからエラーが返されました: {response_body['error']}")
                return response_body

            # OCR結果を軽量化（不要なフィールドを削除）
            if 'words' in response_body:
                simplified_words = []
                for word in response_body['words']:
                    # 必要なフィールドのみを保持
                    simplified_word = {
                        "id": word["id"],
                        "content": word["content"],
                        "points": word["points"]
                    }
                    # 方向情報が必要な場合のみ保持
                    if "direction" in word:
                        simplified_word["direction"] = word["direction"]

                    simplified_words.append(simplified_word)

                response_body['words'] = simplified_words

            # 拡張されたOCR結果を作成
            words = response_body.get('words', [])
            full_text = " ".join([word.get("content", "") for word in words])
            
            # DynamoDB用にDecimal型を使用
            confidence = Decimal('0.95') if words else Decimal('0.0')
            
            enhanced_result = {
                "text": full_text,
                "words": words,
                "confidence": confidence,
                "word_count": len(words)
            }
            
            logger.info(f"OCR完了: {len(words)}単語を検出, テキスト長: {len(full_text)}")
            return enhanced_result

        except Exception as e:
            logger.error(f"SageMakerエンドポイント呼び出しエラー: {str(e)}")
            # エラー情報を返す
            return {
                "error": f"SageMaker endpoint error: {str(e)}", 
                "text": "",
                "words": [],
                "confidence": Decimal('0.0'),
                "word_count": 0
            }

    except Exception as e:
        logger.error(f"OCR処理エラー: {str(e)}")
        return {
            "error": str(e), 
            "text": "",
            "words": [],
            "confidence": Decimal('0.0'),
            "word_count": 0
        }


def process_information_extraction_without_ocr(image_id: str):
    """OCRなしで直接LLMに画像を渡して情報抽出"""
    try:
        # 画像情報を取得
        image_data = get_image(str(image_id) if isinstance(
            image_id, uuid.UUID) else image_id)

        if not image_data:
            logger.error(f"画像 {image_id} が見つかりません")
            return

        s3_key = image_data.get("s3_key")
        # アプリ名を取得（なければデフォルト）
        app_name = image_data.get("app_name", DEFAULT_APP)

        # このアプリ用の抽出フィールド定義を取得
        app_extraction_fields = get_extraction_fields_for_app(app_name)
        field_names = get_field_names_for_app(app_name)

        # カスタムプロンプトを取得
        from app_schema import get_custom_prompt_for_app
        custom_prompt = get_custom_prompt_for_app(app_name)

        logger.info(f"OCRなしモードで情報抽出を開始: {image_id}")
        logger.info(
            f"処理アプリ: {app_name}, フィールド数: {len(app_extraction_fields.get('fields', []))}")

        # S3から画像を取得
        s3_response = s3_client.get_object(
            Bucket=os.getenv("BUCKET_NAME"),
            Key=s3_key
        )
        image_bytes = s3_response['Body'].read()

        # 画像をBase64エンコード
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # OCRなしモード用のプロンプトを作成（fieldsを渡す）
        vision_only_prompt = create_vision_only_extraction_prompt(
            app_extraction_fields.get('fields', []), field_names, custom_prompt
        )

        # LLMに画像のみを渡して情報抽出
        extracted_info = extract_information_from_image_only(
            image_base64, vision_only_prompt, app_extraction_fields
        )

        # 結果をDBに保存（OCRなしモードではマッピング情報は空）
        update_extracted_info(image_id, extracted_info, {})
        update_image_status(image_id, "completed")

        logger.info(f"OCRなしモードでの情報抽出が完了: {image_id}")

    except Exception as e:
        logger.error(f"OCRなしモードでの情報抽出エラー: {str(e)}")
        update_image_status(image_id, "failed")
        raise


def create_vision_only_extraction_prompt(extraction_fields, field_names, custom_prompt=""):
    """OCRなしモード用のプロンプトを作成"""

    # 抽出対象の情報を整理（fieldsはリスト形式）
    extraction_targets = ""
    for field in extraction_fields:
        field_name = field.get("name", "")
        field_type = field.get("type", "text")
        description = field.get("description", "")
        extraction_targets += f"- {field_name} ({field_type}): {description}\n"

    # JSONテンプレートを作成
    json_template = json.dumps(field_names, ensure_ascii=False, indent=2)

    prompt = f"""画像から以下の情報を抽出してください。OCR処理は行わず、画像を直接解析してください。

<extraction_fields>
{extraction_targets}
</extraction_fields>

{f'''
<custom_instructions>
{custom_prompt}
</custom_instructions>
''' if custom_prompt else ''}

<output_format>
以下のJSON形式で出力してください:
{json_template}
</output_format>

注意事項:
- 画像から直接情報を読み取ってください
- 不明な項目は空文字列("")にしてください
- 数値は文字列として出力してください
- 日付は YYYY-MM-DD 形式で出力してください
- JSONのみを出力し、余計な説明は不要です
"""

    return prompt


def extract_json_from_response(response_text):
    """LLMレスポンスからJSONを抽出して解析"""
    try:
        # JSONを含む部分を抽出
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            response_data = json.loads(json_str)

            # 抽出情報を取得（indices以外のキー）
            extracted_info = {k: v for k,
                              v in response_data.items() if k != 'indices'}
            return extracted_info
        else:
            logger.warning("レスポンスからJSONが見つかりませんでした")
            return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析エラー: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"JSON抽出エラー: {str(e)}")
        return {}


def extract_information_from_image_only(image_base64, prompt, extraction_fields):
    """画像のみからLLMを使って情報抽出（既存のconverse_with_model関数を使用）"""
    try:
        # 画像データをデコードしてフォーマットを判定
        image_bytes = base64.b64decode(image_base64)

        # 画像フォーマットを判定
        image_format = "jpeg"  # デフォルト
        if image_bytes.startswith(b'\x89PNG'):
            image_format = "png"
        elif image_bytes.startswith(b'\xff\xd8'):
            image_format = "jpeg"
        elif image_bytes.startswith(b'GIF'):
            image_format = "gif"
        elif image_bytes.startswith(b'WEBP', 8):
            image_format = "webp"

        # システムプロンプト（OCRありモードと統一）
        system_prompts = [{
            "text": "あなたは画像から情報を抽出するアシスタントです。指定されたフィールドに対応する情報を抽出し、JSONフォーマットで返してください。"
        }]

        # メッセージを構築
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    },
                    {
                        "image": {
                            "format": image_format,
                            "source": {
                                "bytes": image_bytes
                            }
                        }
                    }
                ]
            }
        ]

        # 既存のconverse_with_model関数を使用
        response = converse_with_model(messages, system_prompts)

        # レスポンスからテキストを抽出
        response_text = response['output']['message']['content'][0]['text']

        # JSONを抽出して解析
        extracted_data = extract_json_from_response(response_text)

        logger.info(f"OCRなしモードでの抽出完了: {len(extracted_data)} フィールド")
        return extracted_data

    except Exception as e:
        logger.error(f"OCRなしモードでのLLM呼び出しエラー: {str(e)}")
        raise


def process_information_extraction_with_ocr(image_id: str, ocr_result: dict, original_text: str = ""):
    """OCR結果のJSONを使用して情報抽出とマッピングを行う"""
    try:
        # 画像情報を取得
        image_data = get_image(str(image_id) if isinstance(
            image_id, uuid.UUID) else image_id)

        if not image_data:
            logger.error(f"画像 {image_id} が見つかりません")
            return

        s3_key = image_data.get("s3_key")
        # アプリ名を取得（なければデフォルト）
        app_name = image_data.get("app_name", DEFAULT_APP)

        # このアプリ用の抽出フィールド定義を取得
        app_extraction_fields = get_extraction_fields_for_app(app_name)
        field_names = get_field_names_for_app(app_name)

        # カスタムプロンプトを取得
        from app_schema import get_custom_prompt_for_app
        custom_prompt = get_custom_prompt_for_app(app_name)

        logger.info(
            f"処理アプリ: {app_name}, フィールド数: {len(app_extraction_fields.get('fields', []))}")

        # 画像データを取得
        try:
            s3_response = s3_client.get_object(
                Bucket=os.getenv("BUCKET_NAME"),
                Key=s3_key
            )
            image_data = s3_response['Body'].read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # コンテンツタイプからフォーマットを取得
            content_type = s3_response.get('ContentType', 'image/jpeg')
            logger.info(
                f"画像 {image_id} を取得しました: {content_type}, サイズ: {len(image_data)} バイト")
        except Exception as e:
            logger.error(f"画像データ取得エラー: {str(e)}")
            image_base64 = None
            content_type = None

        # Bedrock clientの初期化
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=MODEL_REGION
        )

        # 抽出対象の項目リストを生成
        extraction_fields = []

        def generate_extraction_fields(fields, prefix=""):
            result = []
            for i, field in enumerate(fields):
                display_name = field['display_name']
                field_type = field.get('type', 'string')

                if prefix:
                    field_desc = f"{prefix} > {display_name} ({field_type}型)"
                else:
                    field_desc = f"{display_name} ({field_type}型)"

                result.append(field_desc)

                # map型の場合は子フィールドも追加
                if field_type == "map" and "fields" in field:
                    child_fields = generate_extraction_fields(
                        field["fields"], display_name)
                    result.extend(child_fields)

                # list型の場合はitem内のフィールドも追加
                elif field_type == "list" and "items" in field:
                    items = field["items"]
                    if items.get("type") == "map" and "fields" in items:
                        child_prefix = f"{display_name} (各項目)"
                        child_fields = generate_extraction_fields(
                            items["fields"], child_prefix)
                        result.extend(child_fields)

            return result

        extraction_fields = generate_extraction_fields(
            app_extraction_fields["fields"])
        extraction_targets = "\n".join(
            [f"{i+1}. {field}" for i, field in enumerate(extraction_fields)])

        # JSONのテンプレートを生成
        def generate_field_template(fields, indent=2):
            items = []
            indent_str = " " * indent

            for field in fields:
                if field.get("type") == "string":
                    # 文字列型
                    items.append(
                        f'{indent_str}"{field["name"]}": "{field["display_name"]}の値"')

                elif field.get("type") == "map" and "fields" in field:
                    # マップ型（ネストされたオブジェクト）
                    nested_fields = generate_field_template(
                        field["fields"], indent + 2)
                    items.append(
                        f'{indent_str}"{field["name"]}": {{\n{nested_fields}\n{indent_str}}}')

                elif field.get("type") == "list" and "items" in field:
                    # リスト型
                    items_type = field["items"].get("type")

                    if items_type == "map" and "fields" in field["items"]:
                        # マップのリスト
                        nested_fields = generate_field_template(
                            field["items"]["fields"], indent + 6)
                        items.append(
                            f'{indent_str}"{field["name"]}": [\n{indent_str}  {{\n{nested_fields}\n{indent_str}  }}\n{indent_str}]')
                    else:
                        # 単純なリスト
                        items.append(
                            f'{indent_str}"{field["name"]}": ["値1", "値2"]')

                # 後方互換性のため、古い形式もサポート
                elif field.get("type") == "table" and "columns" in field:
                    # テーブルフィールドの場合
                    columns_json = []
                    for column in field["columns"]:
                        column_name = column["name"]
                        column_display = column["display_name"]
                        columns_json.append(
                            f'{indent_str}      "{column_name}": "{column_display}の値"')

                    joined_columns = ",\n".join(columns_json)
                    items.append(
                        f'{indent_str}"{field["name"]}": [\n{indent_str}  {{\n{joined_columns}\n{indent_str}  }}\n{indent_str}]')
                else:
                    # その他のフィールド（text型など）
                    items.append(
                        f'{indent_str}"{field["name"]}": "{field["display_name"]}の値"')

            return ",\n".join(items)

        json_template = "{\n" + \
            generate_field_template(app_extraction_fields["fields"]) + ",\n"

        # インデックス部分のテンプレートを生成
        def generate_indices_template(fields, indent=4):
            items = []
            indent_str = " " * indent

            for field in fields:
                if field.get("type") == "string":
                    # 文字列型
                    items.append(f'{indent_str}"{field["name"]}": [対応する単語のID]')

                elif field.get("type") == "map" and "fields" in field:
                    # マップ型（ネストされたオブジェクト）
                    nested_indices = generate_indices_template(
                        field["fields"], indent + 2)
                    items.append(
                        f'{indent_str}"{field["name"]}": {{\n{nested_indices}\n{indent_str}}}')

                elif field.get("type") == "list" and "items" in field:
                    # リスト型
                    items_type = field["items"].get("type")

                    if items_type == "map" and "fields" in field["items"]:
                        # マップのリスト
                        nested_indices = []
                        for item_field in field["items"]["fields"]:
                            nested_indices.append(
                                f'{indent_str}      "{item_field["name"]}": [対応する単語のID]')

                        nested_indices_str = ",\n".join(nested_indices)
                        items.append(
                            f'{indent_str}"{field["name"]}": [\n{indent_str}  {{\n{nested_indices_str}\n{indent_str}  }}\n{indent_str}]')
                    else:
                        # 単純なリスト
                        items.append(
                            f'{indent_str}"{field["name"]}": [[対応する単語のID]]')

                # 後方互換性のため、古い形式もサポート
                elif field.get("type") == "table" and "columns" in field:
                    # テーブルフィールドの場合
                    columns_indices = []
                    for column in field["columns"]:
                        column_name = column["name"]
                        columns_indices.append(
                            f'{indent_str}      "{column_name}": [対応する単語のID]')

                    joined_indices = ",\n".join(columns_indices)
                    items.append(
                        f'{indent_str}"{field["name"]}": [\n{indent_str}  {{\n{joined_indices}\n{indent_str}  }}\n{indent_str}]')
                else:
                    # その他のフィールド（text型など）
                    items.append(f'{indent_str}"{field["name"]}": [対応する単語のID]')

            return ",\n".join(items)

        indices_template = '  "indices": {\n' + generate_indices_template(
            app_extraction_fields["fields"]) + "\n  }\n}"

        # 例示用のOCR結果とその抽出例
        example_ocr = {
            "words": [
                {"id": 0, "content": "注文日：2023年5月1日", "points": [
                    [50, 120], [250, 120], [250, 150], [50, 150]]},
                {"id": 1, "content": "委託業務内容：配送業務", "points": [
                    [50, 180], [300, 180], [300, 210], [50, 210]]},
                {"id": 2, "content": "運行日：2023年5月15日", "points": [
                    [50, 240], [250, 240], [250, 270], [50, 270]]},
                {"id": 3, "content": "A001", "points": [
                    [50, 400], [100, 400], [100, 430], [50, 430]]},
                {"id": 4, "content": "東京", "points": [
                    [150, 400], [200, 400], [200, 430], [150, 430]]},
                {"id": 5, "content": "大阪", "points": [
                    [250, 400], [300, 400], [300, 430], [250, 430]]}
            ]
        }

        example_output = {
            "order_date": "2023年5月1日",
            "operation_info": {
                "contract_work": "配送業務",
                "operation_date": "2023年5月15日"
            },
            "shipment_details": [
                {
                    "reception_number": "A001",
                    "destination": "東京",
                    "origin": "大阪",
                    "vehicle_number": "",
                    "fare": ""
                }
            ],
            "indices": {
                "order_date": [0],
                "operation_info": {
                    "contract_work": [1],
                    "operation_date": [2]
                },
                "shipment_details": [
                    {
                        "reception_number": [3],
                        "destination": [4],
                        "origin": [5],
                        "vehicle_number": [],
                        "fare": []
                    }
                ]
            }
        }

        # マルチモーダルでプロンプト作成（画像がある場合）
        if image_base64:
            logger.info(f"画像 {image_id} を含むマルチモーダルプロンプトを作成します")

            # システムプロンプト
            system_prompts = [{
                "text": "あなたはOCR結果から情報を抽出するアシスタントです。指定されたフィールドに対応する情報を抽出し、JSONフォーマットで返してください。"
            }]

            # 画像フォーマットを取得
            image_format = content_type.split(
                '/')[1] if content_type and '/' in content_type else 'jpeg'

            # ユーザーメッセージ
            user_message = {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": image_format,
                            "source": {
                                "bytes": image_data  # バイナリデータを直接使用
                            }
                        }
                    },
                    {
                        "text": f"""
                        次のOCR結果から指定された情報を抽出してください。
                        
                        抽出対象情報には以下の型があります：
                        - string型: 単一の文字列値
                        - map型: 複数のフィールドを持つオブジェクト
                        - list型: 複数の項目を持つ配列
                        
                        抽出した各情報について、対応するOCR結果の単語IDも指定してください。
                        階層構造のデータについては、各フィールドごとにIDを指定してください。
                        リスト型データについては各項目の各フィールドごとにIDを指定してください。
                        
                        情報が見つからない場合は該当項目を空文字列にし、IDは空の配列にしてください。

                        <extraction_example>
                        例えば、以下のようなOCR結果があった場合：
                        {json.dumps(example_ocr, ensure_ascii=False, indent=0)}
                        
                        以下のような抽出結果を期待します：
                        {json.dumps(example_output, ensure_ascii=False, indent=0)}
                        </extraction_example>

                        <extraction_fields>
                        {extraction_targets}
                        </extraction_fields>

                        <ocr_result>
                        {json.dumps(decimal_to_float(ocr_result), ensure_ascii=False, indent=0)}
                        </ocr_result>
                        
                        {f'''
                        <custom_instructions>
                        {custom_prompt}
                        </custom_instructions>
                        ''' if custom_prompt else ''}

                        <output_format>
                        {json_template}{indices_template}
                        </output_format>
                        """
                    }
                ]
            }

            messages = [user_message]
        else:
            # 画像がない場合はテキストのみのプロンプト
            logger.info(f"画像 {image_id} のテキストのみのプロンプトを作成します")

            # システムプロンプト
            system_prompts = [{
                "text": "あなたはOCR結果から情報を抽出するアシスタントです。指定されたフィールドに対応する情報を抽出し、JSONフォーマットで返してください。"
            }]

            # ユーザーメッセージ
            user_message = {
                "role": "user",
                "content": [{
                    "text": f"""
                    次のOCR結果から指定された情報を抽出してください。
                    
                    抽出対象情報には以下の型があります：
                    - string型: 単一の文字列値
                    - map型: 複数のフィールドを持つオブジェクト
                    - list型: 複数の項目を持つ配列
                    
                    抽出した各情報について、対応するOCR結果の単語IDも指定してください。
                    階層構造のデータについては、各フィールドごとにIDを指定してください。
                    リスト型データについては各項目の各フィールドごとにIDを指定してください。
                    
                    情報が見つからない場合は該当項目を空文字列にし、IDは空の配列にしてください。

                    <extraction_example>
                    例えば、以下のようなOCR結果があった場合：
                    {json.dumps(example_ocr, ensure_ascii=False, indent=0)}
                    
                    以下のような抽出結果を期待します：
                    {json.dumps(example_output, ensure_ascii=False, indent=0)}
                    </extraction_example>

                    <extraction_fields>
                    {extraction_targets}
                    </extraction_fields>

                    <ocr_result>
                    {json.dumps(decimal_to_float(ocr_result), ensure_ascii=False, indent=0)}
                    </ocr_result>
                    
                    {f'''
                    <custom_instructions>
                    {custom_prompt}
                    </custom_instructions>
                    ''' if custom_prompt else ''}

                    <output_format>
                    {json_template}{indices_template}
                    </output_format>
                    """
                }]
            }

            messages = [user_message]

        # Bedrock API呼び出し（リトライロジック付き）
        max_retries = 5
        retry_delay = 1  # 初期遅延（秒）

        for attempt in range(max_retries):
            try:
                # メッセージの長さをログに出力
                message_text = ""
                for msg in messages:
                    if "content" in msg:
                        for content_item in msg["content"]:
                            if "text" in content_item:
                                message_text += content_item["text"]

                logger.info(
                    f"画像 {image_id} の情報抽出のためのプロンプト長: {len(message_text)} 文字")

                logger.info(
                    f"画像 {image_id} の情報抽出のためにBedrock APIを呼び出します（試行回数: {attempt+1}/{max_retries}）")

                # Converse APIを使用
                response = converse_with_model(messages, system_prompts)

                # レスポンスを取得
                ai_response = parse_converse_response(response)

                # 成功したらループを抜ける
                break

            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    if attempt < max_retries - 1:
                        # 指数バックオフで待機時間を計算
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            f"スロットリングエラーが発生しました。{wait_time}秒後に再試行します（試行回数: {attempt+1}/{max_retries}）")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"最大再試行回数に達しました。処理を中止します。")
                        raise
                else:
                    # その他のエラーはそのまま再スロー
                    logger.error(f"Bedrock API呼び出しエラー: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"予期しないエラーが発生しました: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"{wait_time}秒後に再試行します（試行回数: {attempt+1}/{max_retries}）")
                    time.sleep(wait_time)
                else:
                    logger.error(f"最大再試行回数に達しました。処理を中止します。")
                    raise

        # JSONを抽出
        extracted_info = {}
        mapping = {}
        try:
            # JSONを含む部分を抽出
            json_match = re.search(r'\{[\s\S]*\}', ai_response)
            if json_match:
                json_str = json_match.group(0)
                response_data = json.loads(json_str)

                # 抽出情報を取得（indices以外のキー）
                extracted_info = {k: v for k,
                                  v in response_data.items() if k != 'indices'}

                # インデックス情報（マッピング）を取得
                if "indices" in response_data:
                    mapping = response_data["indices"]
                    logger.info(
                        f"LLMから直接マッピング情報を取得: {json.dumps(mapping, ensure_ascii=False, indent=0)}")
                else:
                    # インデックス情報がない場合は空のマッピングを設定
                    logger.warning("LLMからマッピング情報を取得できませんでした")
                    mapping = {field_name: [] for field_name in field_names}
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

        # 抽出結果とマッピング情報をデータベースに保存
        update_extracted_info(
            str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
            extracted_info,
            mapping,
            'completed'
        )

        # 画像のステータスも完了に更新
        from database import update_image_status
        update_image_status(
            str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
            "completed"
        )

        logger.info(f"Information extraction completed for image {image_id}")
    except Exception as e:
        logger.error(f"Error in information extraction process: {str(e)}")
        try:
            # エラー時はステータスを失敗に更新
            update_extracted_info(
                str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
                {},
                {},
                'failed'
            )

            # 画像のステータスも失敗に更新
            from database import update_image_status
            update_image_status(
                str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
                "failed"
            )
        except Exception as db_error:
            logger.error(f"Error updating extraction status: {str(db_error)}")


def converse_with_model(messages, system_prompts=None, model_id=None, model_region=None):
    """
    Converse APIを使用してモデルと対話する汎用関数

    Args:
        messages (list): モデルに送信するメッセージのリスト
        system_prompts (list, optional): システムプロンプトのリスト
        model_id (str, optional): 使用するモデルID
        model_region (str, optional): モデルのリージョン

    Returns:
        dict: モデルからの応答
    """
    model_id = model_id or MODEL_ID
    model_region = model_region or MODEL_REGION

    logger.info(f"モデル {model_id} を使用して対話を実行します (リージョン: {model_region})")

    # Bedrock clientの初期化 - read_timeoutを900秒に設定
    from botocore.config import Config
    config = Config(read_timeout=900)  # 900秒 (15分) のタイムアウト設定

    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=model_region,
        config=config
    )

    # 推論パラメータの設定
    inference_config = {
        "temperature": 0.2,
        "maxTokens": 40000
    }

    # モデル固有の追加パラメータ
    additional_model_fields = {}

    try:
        # Converse APIを呼び出し
        logger.info(f"Bedrock APIを呼び出し中 (read_timeout: 900秒)")
        response = bedrock.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts if system_prompts else None,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields if additional_model_fields else None
        )

        # トークン使用量をログに記録
        if 'usage' in response:
            token_usage = response['usage']
            logger.info(f"入力トークン: {token_usage.get('inputTokens', 'N/A')}")
            logger.info(f"出力トークン: {token_usage.get('outputTokens', 'N/A')}")
            logger.info(f"合計トークン: {token_usage.get('totalTokens', 'N/A')}")
            logger.info(f"停止理由: {response.get('stopReason', 'N/A')}")

        return response
    except Exception as e:
        logger.error(f"Converse API呼び出しエラー: {str(e)}")
        raise


def parse_converse_response(response):
    """
    Converse APIのレスポンスからテキスト部分を抽出する

    Args:
        response (dict): Converse APIからのレスポンス

    Returns:
        str: 抽出されたテキスト
    """
    output_message = response['output']['message']
    text = ""

    for content in output_message['content']:
        if 'text' in content:
            text += content['text']

    return text


def generate_schema_fields_from_image(image_data, instructions=None):
    """
    画像からスキーマのフィールド部分のみを生成する関数

    Args:
        image_data (bytes): 画像データ
        instructions (str, optional): スキーマ生成の指示

    Returns:
        dict: 生成されたフィールド定義 {"fields": [...]} の形式
    """
    try:
        # 画像のMIMEタイプを判定
        import imghdr
        image_type = imghdr.what(None, h=image_data)
        content_type = f"image/{image_type}"

        # システムプロンプト
        system_prompts = [{
            "text": """
            あなたはOCR処理された文書から情報抽出のためのフィールド定義を生成するアシスタントです。
            ユーザーが提供する画像を分析し、その文書タイプに適したフィールド構造を生成してください。
            """
        }]

        # フィールド定義の説明
        fields_explanation = """
        フィールド型は主に以下の3種類があります：
        
        1. string型: 単一の文字列値を格納するフィールド（日付、番号、名前など）
        2. map型: 複数の関連フィールドをグループ化するための階層構造（会社情報、住所情報など）
        3. list型: 表形式のデータなど、同じ構造を持つ複数の項目を格納するためのフィールド（明細行、商品リストなど）
        
        基本的には、単一の値は string 型、関連する複数の値をグループ化する場合は map 型、
        表形式のデータ（明細行など）は list 型を使用してください。
        """

        # フィールド定義の例
        fields_definition = """
        フィールドは以下の構造に従って定義してください：
        
        {
          "fields": [
            {
              "name": "フィールド名（英数字、アンダースコア）",
              "display_name": "フィールド表示名（日本語可）",
              "type": "string | map | list"  // フィールドの型
            },
            // map型の場合は子フィールドを定義
            {
              "name": "company_info",
              "display_name": "会社情報",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "会社名",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                }
              ]
            },
            // list型の場合はitemsを定義（表形式のデータ向け）
            {
              "name": "items",
              "display_name": "明細",
              "type": "list",
              "items": {
                "type": "map",
                "fields": [
                  {
                    "name": "description",
                    "display_name": "品目",
                    "type": "string"
                  },
                  {
                    "name": "quantity",
                    "display_name": "数量",
                    "type": "string"
                  }
                ]
              }
            }
          ]
        }
        """

        # 実際のフィールド例
        fields_examples = """
        以下は実際のフィールド定義例です：
        
        1. 請求書処理アプリケーションのフィールド例：
        {
          "fields": [
            {
              "name": "invoice_date",
              "display_name": "請求日",
              "type": "string"
            },
            {
              "name": "invoice_number",
              "display_name": "請求番号",
              "type": "string"
            },
            {
              "name": "company_info",
              "display_name": "会社情報",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "会社名",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                },
                {
                  "name": "phone",
                  "display_name": "電話番号",
                  "type": "string"
                }
              ]
            },
            {
              "name": "items",
              "display_name": "明細",
              "type": "list",
              "items": {
                "type": "map",
                "fields": [
                  {
                    "name": "description",
                    "display_name": "品目",
                    "type": "string"
                  },
                  {
                    "name": "quantity",
                    "display_name": "数量",
                    "type": "string"
                  },
                  {
                    "name": "unit_price",
                    "display_name": "単価",
                    "type": "string"
                  },
                  {
                    "name": "amount",
                    "display_name": "金額",
                    "type": "string"
                  }
                ]
              }
            },
            {
              "name": "total_amount",
              "display_name": "合計金額",
              "type": "string"
            },
            {
              "name": "tax_amount",
              "display_name": "消費税",
              "type": "string"
            }
          ]
        }
        
        2. 輸送伝票処理アプリケーションのフィールド例：
        {
          "fields": [
            {
              "name": "shipping_date",
              "display_name": "出荷日",
              "type": "string"
            },
            {
              "name": "tracking_number",
              "display_name": "追跡番号",
              "type": "string"
            },
            {
              "name": "sender",
              "display_name": "送り主",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "名前",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                },
                {
                  "name": "phone",
                  "display_name": "電話番号",
                  "type": "string"
                }
              ]
            },
            {
              "name": "receiver",
              "display_name": "受取人",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "名前",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                },
                {
                  "name": "phone",
                  "display_name": "電話番号",
                  "type": "string"
                }
              ]
            },
            {
              "name": "items",
              "display_name": "荷物情報",
              "type": "list",
              "items": {
                "type": "map",
                "fields": [
                  {
                    "name": "description",
                    "display_name": "品名",
                    "type": "string"
                  },
                  {
                    "name": "quantity",
                    "display_name": "数量",
                    "type": "string"
                  },
                  {
                    "name": "weight",
                    "display_name": "重量",
                    "type": "string"
                  }
                ]
              }
            }
          ]
        }
        """

        # ユーザーからの指示があれば追加
        instruction_text = ""
        if instructions:
            instruction_text = f"""
            ユーザーからの追加指示：
            {instructions}
            """

        # ユーザーメッセージ
        user_message = {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": content_type.split('/')[1],
                        "source": {
                            "bytes": image_data
                        }
                    }
                },
                {
                    "text": f"""
                    この画像に写っている文書を分析し、情報抽出に適したフィールド定義を生成してください。
                    
                    {instruction_text}
                    
                    {fields_explanation}
                    
                    {fields_definition}
                    
                    {fields_examples}
                    
                    以下の点に注意してフィールドを生成してください：
                    1. 文書の種類（請求書、納品書、輸送伝票など）を特定し、適切なフィールド構造を設計する
                    2. 文書から抽出可能な全ての重要情報をフィールドとして定義する
                    3. 階層構造が必要な場合は適切にmap型を使用する
                    4. 表形式のデータ（明細行など）がある場合はlist型を使用する
                    5. フィールド名は英数字とアンダースコアのみを使用し、日本語は表示名に使用する
                    
                    必ず {{"fields": [...]}} の形式でJSONを出力してください。説明や補足は不要です。
                    """
                }
            ]
        }

        messages = [user_message]

        # Bedrock APIを呼び出し
        response = converse_with_model(messages, system_prompts)

        # レスポンスからテキストを抽出
        fields_text = parse_converse_response(response)

        # JSONテキストからフィールド定義を抽出
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```',
                               fields_text, re.DOTALL)
        if json_match:
            fields_json = json_match.group(1)
        else:
            fields_json = fields_text

        # JSONをパース
        try:
            schema = json.loads(fields_json)

            # スキーマが {"fields": [...]} の形式になっているか確認
            if "fields" not in schema:
                # fieldsキーがない場合は、配列を受け取ったと仮定して包む
                if isinstance(schema, list):
                    schema = {"fields": schema}
                else:
                    # それ以外の場合はエラー
                    raise ValueError("生成されたスキーマに 'fields' キーがありません")

            return schema
        except json.JSONDecodeError as e:
            logger.error(f"フィールド定義のJSONパースエラー: {str(e)}")
            raise ValueError(f"生成されたフィールド定義が有効なJSONではありません: {fields_json}")

    except Exception as e:
        logger.error(f"フィールド生成エラー: {str(e)}")
        raise
