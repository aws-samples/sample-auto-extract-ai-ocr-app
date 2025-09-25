from config import settings
from clients import s3_client, sagemaker_runtime_client
import json
import logging
import base64

from app_schema import get_extraction_fields_for_app, get_field_names_for_app, DEFAULT_APP
from database import get_image, update_extracted_info, update_image_status, update_ocr_result

logger = logging.getLogger(__name__)


def perform_ocr(image_data):
    """画像データに対してOCR処理を実行し、結果を返す（SageMakerエンドポイント使用）"""
    if not settings.ENABLE_OCR:
        raise ValueError("OCR is disabled in this deployment")

    if not settings.SAGEMAKER_ENDPOINT_NAME:
        raise ValueError("SageMaker endpoint not configured")

    try:
        logger.info(
            f"SageMakerエンドポイント {settings.SAGEMAKER_ENDPOINT_NAME} を使用してOCR処理を実行中")

        # 画像をBase64エンコード
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # SageMakerエンドポイントへのリクエストデータを作成
        request_body = {
            "image": image_base64
        }

        # SageMakerエンドポイントを呼び出し
        try:
            # 推論コンポーネントを直接指定してエンドポイントを呼び出し
            response = sagemaker_runtime_client.invoke_endpoint(
                EndpointName=settings.SAGEMAKER_ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps(request_body),
                InferenceComponentName=settings.SAGEMAKER_INFERENCE_COMPONENT_NAME
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

            enhanced_result = {
                "text": full_text,
                "words": words,
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
                "word_count": 0
            }

    except Exception as e:
        logger.error(f"OCR処理エラー: {str(e)}")
        return {
            "error": str(e),
            "text": "",
            "words": [],
            "word_count": 0
        }


def perform_ocr_single_page(s3_key: str):
    """
    単一ページのOCR処理
    """
    try:
        # S3から画像を取得
        bucket_name = settings.BUCKET_NAME
        s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        image_data = s3_response['Body'].read()

        # perform_ocr関数を使用
        ocr_result = perform_ocr(image_data)

        # エラーチェック
        if "error" in ocr_result:
            raise ValueError(f"OCR処理エラー: {ocr_result['error']}")

        # perform_ocr関数は既にDecimal型を返すので、そのまま使用
        return {
            "words": ocr_result.get("words", []),
            "text": ocr_result.get("text", "")
        }

    except Exception as e:
        logger.error(f"単一ページOCR処理エラー: {str(e)}")
        raise


def perform_ocr_multipage(image_id: str):
    """
    複数ページのOCR処理
    """
    try:
        logger.info(f"複数ページOCR処理を開始: {image_id}")

        # 画像データを取得
        image_data = get_image(image_id)
        converted_s3_keys = image_data.get("converted_s3_key")

        if not converted_s3_keys or not isinstance(converted_s3_keys, list):
            raise ValueError("複数ページの変換済み画像が見つかりません")

        ocr_results = []

        # 各ページをOCR処理
        for i, s3_key in enumerate(converted_s3_keys):
            try:
                logger.info(
                    f"ページ {i+1}/{len(converted_s3_keys)} OCR処理中: {s3_key}")

                # 単一ページOCR処理
                page_ocr_result = perform_ocr_single_page(s3_key)

                # ページ情報を追加
                page_result = {
                    "page": i + 1,
                    "words": page_ocr_result.get("words", []),
                    "text": page_ocr_result.get("text", "")
                }

                ocr_results.append(page_result)
                logger.info(f"ページ {i+1} OCR完了")

            except Exception as e:
                logger.error(f"ページ {i+1} OCR処理エラー: {str(e)}")
                # エラーページも記録（空の結果として）
                ocr_results.append({
                    "page": i + 1,
                    "words": [],
                    "text": "",
                    "error": str(e)
                })
                continue

        # 複数ページOCR結果を保存
        save_multipage_ocr_result(image_id, ocr_results)

        logger.info(f"複数ページOCR処理完了: {image_id}")
        return ocr_results

    except Exception as e:
        logger.error(f"複数ページOCR処理エラー: {str(e)}")
        raise


def perform_ocr_individual_page(image_id: str):
    """個別ページのOCR処理"""
    try:
        logger.info(f"個別ページ処理を実行: {image_id}")

        # 画像情報を取得
        image_data = get_image(image_id)
        if not image_data:
            raise ValueError(f"Image not found: {image_id}")

        # S3から画像をダウンロード
        s3_key = image_data.get("s3_key")
        if isinstance(s3_key, list):
            s3_key = s3_key[0]  # リストの場合は最初の要素

        from clients import s3_client
        from config import settings
        s3_response = s3_client.get_object(
            Bucket=settings.BUCKET_NAME, Key=s3_key)
        image_bytes = s3_response['Body'].read()

        # OCR処理を行う
        ocr_result = perform_ocr(image_bytes)

        # OCR結果にエラーがある場合の処理
        if "error" in ocr_result:
            logger.error(f"OCR処理でエラーが発生: {ocr_result['error']}")
            update_image_status(image_id, "failed")
            return

        logger.info(
            f"Successfully processed {len(ocr_result.get('words', []))} words for image {image_id}")

        # OCR結果をテキストとして取得
        ocr_text = ocr_result.get("text", "")
        if not ocr_text:
            # フォールバック: wordsから結合
            ocr_text = "\n".join([word.get("content", "")
                                 for word in ocr_result.get("words", [])])

        # DynamoDBにOCR結果を保存
        logger.info(f"Saving OCR results for image {image_id}")
        update_ocr_result(image_id, ocr_result, "processing")

    except Exception as e:
        logger.error(f"個別ページOCR処理エラー: {str(e)}")
        update_image_status(image_id, "failed")
        raise


def perform_ocr_single_image(image_id: str):
    """単一画像のOCR処理"""
    try:
        logger.info(f"単一画像処理を実行: {image_id}")

        # 画像情報を取得
        image_data = get_image(image_id)
        if not image_data:
            raise ValueError(f"Image not found: {image_id}")

        # S3から画像をダウンロード
        s3_key = image_data.get("converted_s3_key") or image_data.get("s3_key")
        if isinstance(s3_key, list):
            s3_key = s3_key[0]

        from clients import s3_client
        from config import settings
        s3_response = s3_client.get_object(
            Bucket=settings.BUCKET_NAME, Key=s3_key)
        image_bytes = s3_response['Body'].read()

        # OCR処理を行う
        ocr_result = perform_ocr(image_bytes)

        # OCR結果にエラーがある場合の処理
        if "error" in ocr_result:
            logger.error(f"OCR処理でエラーが発生: {ocr_result['error']}")
            update_image_status(image_id, "failed")
            return

        logger.info(
            f"Successfully processed {len(ocr_result.get('words', []))} words for image {image_id}")

        # OCR結果をテキストとして取得
        ocr_text = ocr_result.get("text", "")
        if not ocr_text:
            # フォールバック: wordsから結合
            ocr_text = "\n".join([word.get("content", "")
                                 for word in ocr_result.get("words", [])])

        # DynamoDBにOCR結果を保存
        logger.info(f"Saving OCR results for image {image_id}")
        update_ocr_result(image_id, ocr_result, "processing")

    except Exception as e:
        logger.error(f"単一画像OCR処理エラー: {str(e)}")
        update_image_status(image_id, "failed")
        raise


def save_multipage_ocr_result(image_id: str, ocr_results: list):
    """
    複数ページOCR結果を保存
    """
    try:
        from decimal import Decimal

        # 統合OCR結果を作成（全ページ通してユニークなIDを付与）
        all_words = []
        global_word_id = 0  # 全ページ通してのユニークID

        for page_result in ocr_results:
            page_words = page_result.get("words", [])
            # 各単語にページ情報を追加し、ユニークなIDを再付与
            for word in page_words:
                word["page"] = page_result["page"]
                word["id"] = global_word_id  # 全ページ通してユニークなIDを付与
                global_word_id += 1
            all_words.extend(page_words)

        # ページ別結果のIDも更新（参照用）
        updated_pages = []
        for page_result in ocr_results:
            updated_page = page_result.copy()
            # このページの単語のIDを更新済みのものに合わせる
            page_words = []
            for word in page_result.get("words", []):
                # all_wordsから対応する単語を見つける（ページとcontentで照合）
                for updated_word in all_words:
                    if (updated_word.get("page") == page_result["page"] and
                        updated_word.get("content") == word.get("content") and
                            updated_word.get("points") == word.get("points")):
                        page_words.append(updated_word)
                        break
            updated_page["words"] = page_words
            updated_pages.append(updated_page)

        # Float型をDecimal型に変換
        def convert_floats_to_decimal(obj):
            if isinstance(obj, dict):
                return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats_to_decimal(item) for item in obj]
            elif isinstance(obj, float):
                return Decimal(str(obj))
            else:
                return obj

        # 統合結果を保存（Float型をDecimal型に変換）
        combined_result = convert_floats_to_decimal({
            "words": all_words,
            "pages": updated_pages,  # ID更新済みのページ別結果も保存
            "total_pages": len(ocr_results)
        })

        update_ocr_result(image_id, combined_result, "completed")
        logger.info(
            f"複数ページOCR結果保存完了: {image_id}, 総単語数: {len(all_words)}, ID範囲: 0-{global_word_id-1}")

    except Exception as e:
        logger.error(f"複数ページOCR結果保存エラー: {str(e)}")
        raise
