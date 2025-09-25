from utils.prompts import (
    create_single_with_ocr_prompt, create_single_without_ocr_prompt,
    create_multi_with_ocr_prompt, create_multi_without_ocr_prompt
)
from utils.helpers import safe_get_from_dynamo_data, float_to_decimal
from config import settings
from utils.bedrock import call_bedrock, call_bedrock_with_retry, parse_converse_response, extract_json_from_response, parse_extraction_response
from clients import s3_client
from utils.template import generate_unified_template
import logging
import base64
import uuid

from app_schema import get_app_schema, get_extraction_fields_for_app, get_field_names_for_app, get_custom_prompt_for_app, DEFAULT_APP
from database import get_image, update_extracted_info, update_image_status

logger = logging.getLogger(__name__)


def extract_information_from_single_image_with_ocr(image_id: str):
    """
    単一画像+OCR結果での情報抽出（統一版：関数内でOCR結果取得）
    """
    try:
        logger.info(f"単一画像情報抽出を開始: {image_id}")

        # 画像情報を取得
        image_data = get_image(str(image_id) if isinstance(
            image_id, uuid.UUID) else image_id)

        if not image_data:
            logger.error(f"画像 {image_id} が見つかりません")
            return

        # OCR結果を関数内で取得（統一）
        ocr_result = image_data.get("ocr_result", {})
        ocr_text = ocr_result.get("text", "")
        if not ocr_text:
            # フォールバック: wordsから結合
            ocr_text = "\n".join(
                [word.get("content", "") for word in ocr_result.get("words", [])])

        # converted_s3_keyを使用（リスト形式に対応）
        converted_s3_keys = image_data.get("converted_s3_key", [])
        if not converted_s3_keys:
            # フォールバックとしてs3_keyを確認
            s3_keys = image_data.get("s3_key", [])
            if isinstance(s3_keys, list) and s3_keys:
                converted_s3_keys = s3_keys
            elif isinstance(s3_keys, str):
                converted_s3_keys = [s3_keys]
            else:
                logger.error(f"S3キーが見つかりません: {image_id}")
                update_image_status(
                    str(image_id) if isinstance(
                        image_id, uuid.UUID) else image_id,
                    "failed"
                )
                return

        # リストから最初のキーを取得
        if isinstance(converted_s3_keys, list):
            s3_key = converted_s3_keys[0] if converted_s3_keys else None
        else:
            s3_key = converted_s3_keys

        if not s3_key:
            logger.error(f"有効なS3キーが見つかりません: {image_id}")
            update_image_status(
                str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
                "failed"
            )
            return

        # アプリ名を取得（なければデフォルト）
        app_name = image_data.get("app_name", DEFAULT_APP)

        logger.info(f"使用するS3キー: {s3_key}")

        # アプリケーションスキーマを取得
        app_schema = get_app_schema(app_name)
        if not app_schema:
            logger.error(f"App schema not found: {app_name}")
            return

        # このアプリ用の抽出フィールド定義を取得
        app_extraction_fields = get_extraction_fields_for_app(app_name)
        field_names = get_field_names_for_app(app_name)

        # カスタムプロンプトを取得
        custom_prompt = get_custom_prompt_for_app(app_name)

        logger.info(
            f"処理アプリ: {app_name}, フィールド数: {len(app_extraction_fields.get('fields', []))}")

        # 画像データを取得
        try:
            s3_response = s3_client.get_object(
                Bucket=settings.BUCKET_NAME,
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
            update_image_status(
                str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
                "failed"
            )
            return

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

        # JSONテンプレートとindicesテンプレートを生成（templateモジュールを使用）
        unified_template = generate_unified_template(app_extraction_fields)

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

        # プロンプト作成
        prompt = create_single_with_ocr_prompt(
            extraction_targets, unified_template, example_ocr, example_output,
            ocr_result, custom_prompt
        )

        # メッセージ構築
        # システムプロンプト
        system_prompts = [{
            "text": "あなたはOCR結果から情報を抽出するアシスタントです。指定されたフィールドに対応する情報を抽出し、JSONフォーマットで返してください。"
        }]

        # マルチモーダルでプロンプト作成（画像がある場合）
        if image_base64:
            logger.info("画像を含むマルチモーダルプロンプトを作成します")

            # 画像フォーマットを取得
            image_format = content_type.split(
                '/')[1] if content_type and '/' in content_type else 'jpeg'

            # ユーザーメッセージ
            messages = [{
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
                        "text": prompt
                    }
                ]
            }]
        else:
            # 画像がない場合はテキストのみのプロンプト
            logger.info("テキストのみのプロンプトを作成します")

            # ユーザーメッセージ
            messages = [{
                "role": "user",
                "content": [{
                    "text": prompt
                }]
            }]

        # Bedrock API呼び出し（リトライロジック付き）
        response = call_bedrock_with_retry(messages, system_prompts)

        # レスポンスからテキストを抽出
        ai_response = parse_converse_response(response)

        # JSONを抽出してマッピング情報を処理
        extracted_info, mapping = parse_extraction_response(
            ai_response, field_names)

        # float値をDecimal型に変換してからデータベースに保存
        extracted_info = float_to_decimal(extracted_info)
        mapping = float_to_decimal(mapping)

        # 抽出結果とマッピング情報をデータベースに保存
        update_extracted_info(
            str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
            extracted_info,
            mapping,
            'completed'
        )

        # 画像のステータスも完了に更新
        update_image_status(
            str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
            "completed"
        )

        logger.info(f"単一画像情報抽出完了: {image_id}")

    except Exception as e:
        logger.error(f"単一画像情報抽出エラー: {str(e)}")
        try:
            # エラー時はステータスを失敗に更新
            update_extracted_info(
                str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
                {},
                {},
                'failed'
            )

            # 画像のステータスも失敗に更新
            update_image_status(
                str(image_id) if isinstance(image_id, uuid.UUID) else image_id,
                "failed"
            )
        except Exception as db_error:
            logger.error(f"Error updating extraction status: {str(db_error)}")
        raise


def extract_information_from_multi_images_with_ocr(image_id: str):
    """
    複数画像+OCR結果での情報抽出
    """
    try:
        logger.info(f"複数画像情報抽出を開始: {image_id}")

        # 画像データを取得
        image_data = get_image(image_id)
        app_name = image_data.get("app_name", DEFAULT_APP)

        # アプリケーションスキーマを取得
        app_schema = get_app_schema(app_name)
        if not app_schema:
            logger.error(f"App schema not found: {app_name}")
            return

        # このアプリ用の抽出フィールド定義を取得
        app_extraction_fields = get_extraction_fields_for_app(app_name)
        field_names = get_field_names_for_app(app_name)

        # カスタムプロンプトを取得
        custom_prompt = get_custom_prompt_for_app(app_name)

        logger.info(f"複数画像情報抽出を開始: {image_id}")
        logger.info(
            f"処理アプリ: {app_name}, フィールド数: {len(app_extraction_fields.get('fields', []))}")

        # 画像データとOCR結果を取得
        converted_s3_keys = safe_get_from_dynamo_data(
            image_data, "converted_s3_key", [])

        if not converted_s3_keys:
            raise ValueError("変換済み画像が見つかりません")

        # リスト形式でない場合は単一画像として扱う
        if not isinstance(converted_s3_keys, list):
            converted_s3_keys = [converted_s3_keys]

        # OCR結果を取得
        ocr_results = get_multipage_ocr_results(image_id)

        if not ocr_results:
            raise ValueError("OCR結果が見つかりません")

        # 抽出指示を作成
        instructions = f"以下のスキーマに従って、文書から情報を抽出してください。"

        # プロンプト生成（カスタムプロンプトを渡す）
        prompt = create_multi_with_ocr_prompt(
            ocr_results, app_extraction_fields, instructions, custom_prompt)

        # 複数画像を取得
        page_images = []
        for s3_key in converted_s3_keys:
            try:
                image_bytes = get_s3_object_bytes(s3_key)
                page_images.append(image_bytes)
            except Exception as s3_error:
                logger.error(f"S3画像取得エラー {s3_key}: {str(s3_error)}")
                continue

        if not page_images:
            raise ValueError("画像データを取得できませんでした")

        logger.info(f"画像数: {len(page_images)}, OCRページ数: {len(ocr_results)}")

        # システムプロンプトを設定
        system_prompts = [{
            "text": "あなたは複数ページの文書から情報を抽出するアシスタントです。指定されたフィールドに対応する情報を抽出し、純粋なJSONオブジェクトのみを返してください。説明文、コメント、マークダウン記法は一切使用しないでください。"
        }]

        # メッセージコンテンツを構築
        content = [{"text": prompt}]

        # 各ページの画像を追加
        for i, image_bytes in enumerate(page_images):
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": image_bytes}
                }
            })

        # メッセージを構築
        messages = [{"role": "user", "content": content}]

        # converse_with_model関数を使用してBedrock呼び出し
        response = call_bedrock(messages, system_prompts)

        # レスポンスを解析
        response_text = parse_converse_response(response)

        # parse_extraction_responseを使用して統一的に解析
        extracted_info, mapping = parse_extraction_response(
            response_text, field_names)

        # float値をDecimal型に変換してからデータベースに保存
        extracted_info = float_to_decimal(extracted_info)
        mapping = float_to_decimal(mapping)

        # 結果を保存（マッピング情報も含める）
        update_extracted_info(image_id, extracted_info, mapping, "completed")
        update_image_status(image_id, "completed")

        logger.info(f"複数画像情報抽出完了: {image_id}")

    except Exception as e:
        logger.error(f"複数画像情報抽出エラー: {str(e)}")
        update_image_status(image_id, "failed")
        raise


def extract_information_from_multi_images_without_ocr(image_id: str):
    """
    OCRなしでの複数画像情報抽出
    """
    try:
        logger.info(f"OCRなし複数画像情報抽出を開始: {image_id}")

        # 画像情報を取得
        image_data = get_image(str(image_id) if isinstance(
            image_id, uuid.UUID) else image_id)

        if not image_data:
            logger.error(f"画像 {image_id} が見つかりません")
            return

        # アプリ名を取得（なければデフォルト）
        app_name = image_data.get("app_name", DEFAULT_APP)
        converted_s3_keys = image_data.get("converted_s3_key", [])

        # 複数画像の検証
        if not isinstance(converted_s3_keys, list) or len(converted_s3_keys) <= 1:
            logger.error(f"複数画像が見つかりません: {converted_s3_keys}")
            update_image_status(image_id, "failed")
            return

        # このアプリ用の抽出フィールド定義を取得
        app_extraction_fields = get_extraction_fields_for_app(app_name)
        field_names = get_field_names_for_app(app_name)

        # カスタムプロンプトを取得
        custom_prompt = get_custom_prompt_for_app(app_name)

        logger.info(f"OCRなし複数画像モードで情報抽出を開始: {image_id}")
        logger.info(
            f"処理アプリ: {app_name}, フィールド数: {len(app_extraction_fields.get('fields', []))}, 画像数: {len(converted_s3_keys)}")

        # 複数画像をS3から取得
        images_data = []
        for i, s3_key in enumerate(converted_s3_keys):
            try:
                s3_response = s3_client.get_object(
                    Bucket=settings.BUCKET_NAME,
                    Key=s3_key
                )
                image_bytes = s3_response['Body'].read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                # コンテンツタイプを取得
                content_type = s3_response.get('ContentType', 'image/jpeg')

                images_data.append({
                    'base64': image_base64,
                    'bytes': image_bytes,
                    'content_type': content_type,
                    'page': i + 1
                })

                logger.info(
                    f"画像 {i+1}/{len(converted_s3_keys)} を取得: {content_type}, サイズ: {len(image_bytes)} バイト")

            except Exception as e:
                logger.error(f"画像データ取得エラー (ページ {i+1}): {str(e)}")
                update_image_status(image_id, "failed")
                return

        # OCRなし複数画像用のプロンプトを作成
        vision_multiimage_prompt = create_multi_without_ocr_prompt(
            app_extraction_fields.get('fields', []), field_names, custom_prompt
        )

        # LLMに複数画像を渡して情報抽出
        try:
            # メッセージコンテンツを構築
            content = []

            # 各画像を追加
            for img_data in images_data:
                # 画像フォーマットを取得
                content_type = img_data['content_type']
                image_format = content_type.split(
                    '/')[1] if content_type and '/' in content_type else 'jpeg'

                content.append({
                    "image": {
                        "format": image_format,
                        "source": {
                            "bytes": img_data['bytes']
                        }
                    }
                })

            # プロンプトテキストを最後に追加
            content.append({
                "text": vision_multiimage_prompt
            })

            # メッセージを構築
            messages = [{
                "role": "user",
                "content": content
            }]

            # システムプロンプト
            system_prompts = [{
                "text": "あなたは複数の画像から情報を抽出するアシスタントです。画像を直接解析して、指定されたフィールドに対応する情報を抽出し、JSONフォーマットで返してください。"
            }]

            logger.info(f"複数画像（{len(images_data)}枚）でLLM呼び出しを開始")

            # Bedrock APIを呼び出し
            response = call_bedrock(messages, system_prompts)
            response_text = parse_converse_response(response)

            logger.info(f"LLMレスポンス取得完了: {len(response_text)} 文字")

            # JSONを抽出
            extracted_info = extract_json_from_response(response_text)
            if not extracted_info:
                logger.warning("JSONの抽出に失敗しました")
                extracted_info = {
                    "error": "Failed to extract JSON from response"}

        except Exception as e:
            logger.error(f"複数画像OCRなしモードでのLLM呼び出しエラー: {str(e)}")
            extracted_info = {"error": str(e)}

        # float値をDecimal型に変換してからデータベースに保存
        extracted_info = float_to_decimal(extracted_info)

        # 結果をDBに保存（OCRなしモードではマッピング情報は空）
        update_extracted_info(image_id, extracted_info, {})
        update_image_status(image_id, "completed")

        logger.info(f"OCRなし複数画像情報抽出完了: {image_id}")

    except Exception as e:
        logger.error(f"OCRなし複数画像情報抽出エラー: {str(e)}")
        update_image_status(image_id, "failed")
        raise


def extract_information_from_single_image_without_ocr(image_id: str):
    """
    OCRなしでの情報抽出（画像のみ → LLM）
    """
    try:
        logger.info(f"OCRなし情報抽出を開始: {image_id}")

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
        custom_prompt = get_custom_prompt_for_app(app_name)

        logger.info(f"OCRなしモードで情報抽出を開始: {image_id}")
        logger.info(
            f"処理アプリ: {app_name}, フィールド数: {len(app_extraction_fields.get('fields', []))}")

        # S3から画像を取得
        s3_response = s3_client.get_object(
            Bucket=settings.BUCKET_NAME,
            Key=s3_key
        )
        image_bytes = s3_response['Body'].read()

        # 画像をBase64エンコード
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # OCRなしモード用のプロンプトを作成（fieldsを渡す）
        vision_only_prompt = create_single_without_ocr_prompt(
            app_extraction_fields.get('fields', []), field_names, custom_prompt
        )

        # LLMに画像のみを渡して情報抽出
        try:
            # 画像データをデコードしてフォーマットを判定
            image_data = base64.b64decode(image_base64)

            # 画像フォーマットを判定（簡易版）
            if image_data.startswith(b'\xff\xd8'):
                image_format = "jpeg"
            elif image_data.startswith(b'\x89PNG'):
                image_format = "png"
            else:
                image_format = "jpeg"  # デフォルト

            # メッセージを構築
            messages = [{
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": image_format,
                            "source": {
                                "bytes": image_data
                            }
                        }
                    },
                    {
                        "text": vision_only_prompt
                    }
                ]
            }]

            # システムプロンプト
            system_prompts = [{
                "text": "あなたは画像から情報を抽出するアシスタントです。画像を直接解析して、指定されたフィールドに対応する情報を抽出し、JSONフォーマットで返してください。"
            }]

            logger.info("OCRなしモードでLLM呼び出しを開始")

            # Bedrock APIを呼び出し
            response = call_bedrock(messages, system_prompts)
            response_text = parse_converse_response(response)

            logger.info(f"LLMレスポンス取得完了: {len(response_text)} 文字")

            # JSONを抽出
            extracted_info = extract_json_from_response(response_text)
            if not extracted_info:
                logger.warning("JSONの抽出に失敗しました")
                extracted_info = {
                    "error": "Failed to extract JSON from response"}

        except Exception as e:
            logger.error(f"OCRなしモードでのLLM呼び出しエラー: {str(e)}")
            extracted_info = {"error": str(e)}

        # float値をDecimal型に変換してからデータベースに保存
        extracted_info = float_to_decimal(extracted_info)

        # 結果をDBに保存（OCRなしモードではマッピング情報は空）
        update_extracted_info(image_id, extracted_info, {})
        update_image_status(image_id, "completed")

        logger.info(f"OCRなし情報抽出完了: {image_id}")

    except Exception as e:
        logger.error(f"OCRなし情報抽出エラー: {str(e)}")
        update_image_status(image_id, "failed")
        raise


def get_multipage_ocr_results(image_id: str) -> list:
    """複数ページOCR結果を取得"""
    try:
        image_data = get_image(image_id)
        ocr_result = safe_get_from_dynamo_data(image_data, "ocr_result", {})

        # 複数ページOCR結果を取得
        pages_results = safe_get_from_dynamo_data(ocr_result, "pages", [])

        # pages_resultsがリストの場合
        if isinstance(pages_results, list):
            processed_pages = []
            for i, page_result in enumerate(pages_results):
                try:
                    if isinstance(page_result, dict):
                        processed_pages.append(page_result)
                    else:
                        logger.warning(
                            f"ページ {i} の結果が辞書形式ではありません: {type(page_result)}")
                except Exception as page_error:
                    logger.error(f"ページ {i} の処理エラー: {str(page_error)}")
                    continue

            if processed_pages:
                return processed_pages

        # pages_resultsが辞書の場合は単一ページとして扱う
        elif isinstance(pages_results, dict):
            return [pages_results]

        # 従来形式の場合は単一ページとして扱う
        words = safe_get_from_dynamo_data(ocr_result, "words", [])

        # wordsがリストでない場合は空リストに
        if not isinstance(words, list):
            logger.warning(f"単語データがリスト形式ではありません: {type(words)}")
            words = []

        return [{"page": 1, "words": words}]

    except Exception as e:
        logger.error(f"複数ページOCR結果取得エラー: {str(e)}")
        return []


def get_s3_object_bytes(s3_key: str) -> bytes:
    """S3から画像バイトデータを取得"""
    try:
        bucket_name = settings.BUCKET_NAME
        s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        return s3_response['Body'].read()
    except Exception as e:
        logger.error(f"S3オブジェクト取得エラー: {s3_key}, {str(e)}")
        raise
