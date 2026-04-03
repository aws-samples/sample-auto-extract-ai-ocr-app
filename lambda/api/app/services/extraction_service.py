import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from repositories import (
    get_image, update_extracted_info,
    update_image_status, get_extraction_fields_for_app,
    get_custom_prompt_for_app,
    get_app_display_name, update_verification_status
)
from schemas import ExtractionRequest
from config import settings
from background import BackgroundTaskExtension
from utils import decimal_to_float
from utils.bedrock import parse_converse_response, extract_json_from_response
from domains.schema_fields import extract_field_names
from clients import s3_client
from clients.bedrock import call_bedrock, call_bedrock_with_retry
from domains.extraction_engine import (
    build_single_image_with_ocr_request,
    build_multi_images_with_ocr_request,
    build_multi_images_without_ocr_request,
    build_single_image_without_ocr_request,
    parse_extraction_response,
    finalize_extraction_result,
)
from services.parent_status import sync_parent_status

logger = logging.getLogger(__name__)


def get_multipage_ocr_results(image_id: str) -> list:
    """複数ページOCR結果を取得する（repository 経由）"""
    try:
        image_data = get_image(image_id)
        ocr_result = image_data.get("ocr_result", {}) if isinstance(image_data, dict) else {}

        # 複数ページOCR結果を取得
        pages_results = ocr_result.get("pages", []) if isinstance(ocr_result, dict) else []

        # pages_resultsがリストの場合
        if isinstance(pages_results, list):
            processed_pages = []
            for i, page_result in enumerate(pages_results):
                try:
                    if isinstance(page_result, dict):
                        processed_pages.append(page_result)
                    else:
                        logger.warning(f"ページ {i} の結果が辞書形式ではありません: {type(page_result)}")
                except Exception as page_error:
                    logger.error(f"ページ {i} の処理エラー: {str(page_error)}")
                    continue
            if processed_pages:
                return processed_pages
        # pages_resultsが辞書の場合は単一ページとして扱う
        elif isinstance(pages_results, dict):
            return [pages_results]

        # 従来形式の場合は単一ページとして扱う
        words = ocr_result.get("words", []) if isinstance(ocr_result, dict) else []
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
        s3_response = s3_client.get_object(Bucket=settings.BUCKET_NAME, Key=s3_key)
        return s3_response['Body'].read()
    except Exception as e:
        logger.error(f"S3オブジェクト取得エラー: {s3_key}, {str(e)}")
        raise


# ===== 抽出プロセッサークラス =====

class InformationExtractor(ABC):
    """情報抽出の基底クラス"""

    def __init__(self, image_id: str, image_data: dict):
        self.image_id = image_id
        self.image_data = image_data

    @abstractmethod
    def extract(self) -> None:
        """情報抽出を実行"""
        pass


class MultiImageExtractor(InformationExtractor):
    """複数画像情報抽出プロセッサー"""

    def extract(self) -> None:
        """複数画像からの情報抽出を実行"""
        logger.info(f"複数画像での情報抽出を実行: {self.image_id}")

        try:
            image_data = get_image(self.image_id)
            if not image_data:
                logger.error(f"画像 {self.image_id} が見つかりません")
                update_image_status(self.image_id, "failed")
                raise ValueError(f"画像 {self.image_id} が見つかりません")

            app_name = image_data.get("app_name")
            if not app_name:
                logger.error(f"app_name not found for image {self.image_id}")
                raise ValueError(f"app_name not found for image {self.image_id}")
            
            app_extraction_fields = get_extraction_fields_for_app(app_name)
            field_names = extract_field_names(app_extraction_fields.get("fields", []))
            custom_prompt = get_custom_prompt_for_app(app_name)

            logger.info(
                f"処理アプリ: {app_name}, フィールド数: {len(app_extraction_fields.get('fields', []))}")

            converted_s3_keys = image_data.get("converted_s3_key", [])

            if not converted_s3_keys:
                raise ValueError("変換済み画像が見つかりません")

            if not isinstance(converted_s3_keys, list):
                converted_s3_keys = [converted_s3_keys]

            # S3から画像データを取得（OCR有無に関わらず必要）
            page_images = []
            content_type = 'image/jpeg'
            for s3_key in converted_s3_keys:
                try:
                    s3_response = s3_client.get_object(
                        Bucket=settings.BUCKET_NAME,
                        Key=s3_key
                    )
                    image_bytes = s3_response['Body'].read()
                    page_images.append(image_bytes)
                    if len(page_images) == 1:
                        content_type = s3_response.get(
                            'ContentType', 'image/jpeg')
                except Exception as s3_error:
                    logger.error(f"S3画像取得エラー {s3_key}: {str(s3_error)}")
                    continue

            if not page_images:
                raise ValueError("画像データを取得できませんでした")

            if settings.ENABLE_OCR:
                ocr_results = get_multipage_ocr_results(self.image_id)

                if not ocr_results:
                    raise ValueError("OCR結果が見つかりません")

                messages, system_prompts = build_multi_images_with_ocr_request(
                    page_images=page_images,
                    content_type=content_type,
                    ocr_results=ocr_results,
                    app_extraction_fields=app_extraction_fields,
                    custom_prompt=custom_prompt
                )
                response = call_bedrock(messages, system_prompts)
                ai_response = parse_converse_response(response)
                extracted_info, mapping = parse_extraction_response(ai_response, field_names)
                result = finalize_extraction_result(extracted_info, mapping)
            else:
                logger.info("OCR無効: without_ocrモードで複数画像情報抽出を実行")
                images_data = [
                    {'bytes': img, 'content_type': content_type}
                    for img in page_images
                ]
                messages, system_prompts = build_multi_images_without_ocr_request(
                    images_data=images_data,
                    app_extraction_fields=app_extraction_fields,
                    field_names=field_names,
                    custom_prompt=custom_prompt
                )
                response = call_bedrock(messages, system_prompts)
                ai_response = parse_converse_response(response)
                extracted_info = extract_json_from_response(ai_response)
                if not extracted_info:
                    extracted_info = {"error": "Failed to extract JSON from response"}
                result = finalize_extraction_result(extracted_info)
                result["mapping"] = {}

            update_extracted_info(
                self.image_id,
                result["extracted_info"],
                result.get("mapping", {}),
                'completed'
            )
            update_image_status(self.image_id, "completed")

            logger.info(f"複数画像情報抽出完了: {self.image_id}")

        except Exception as e:
            logger.error(f"複数画像情報抽出エラー: {str(e)}")
            update_image_status(self.image_id, "failed")
            raise


class SingleImageExtractor(InformationExtractor):
    """単一画像情報抽出プロセッサー"""

    def extract(self) -> None:
        """単一画像からの情報抽出を実行"""
        logger.info(f"単一画像での情報抽出を実行: {self.image_id}")

        try:
            image_data = get_image(self.image_id)
            if not image_data:
                logger.error(f"画像 {self.image_id} が見つかりません")
                update_image_status(self.image_id, "failed")
                raise ValueError(f"画像 {self.image_id} が見つかりません")

            app_name = image_data.get("app_name")
            if not app_name:
                logger.error(f"app_name not found for image {self.image_id}")
                raise ValueError(f"app_name not found for image {self.image_id}")
            
            app_extraction_fields = get_extraction_fields_for_app(app_name)
            field_names = extract_field_names(app_extraction_fields.get("fields", []))
            custom_prompt = get_custom_prompt_for_app(app_name)

            logger.info(
                f"処理アプリ: {app_name}, フィールド数: {len(app_extraction_fields.get('fields', []))}")

            converted_s3_keys = image_data.get("converted_s3_key", [])

            if not converted_s3_keys:
                raise ValueError("変換済み画像が見つかりません")

            s3_key = converted_s3_keys[0] if isinstance(
                converted_s3_keys, list) else converted_s3_keys

            if not s3_key:
                raise ValueError("有効なS3キーが見つかりません")

            s3_response = s3_client.get_object(
                Bucket=settings.BUCKET_NAME,
                Key=s3_key
            )
            image_bytes = s3_response['Body'].read()
            content_type = s3_response.get('ContentType', 'image/jpeg')

            if settings.ENABLE_OCR:
                ocr_result = image_data.get("ocr_result", {})
                messages, system_prompts = build_single_image_with_ocr_request(
                    image_data=image_bytes,
                    content_type=content_type,
                    ocr_result=ocr_result,
                    app_extraction_fields=app_extraction_fields,
                    custom_prompt=custom_prompt
                )
                response = call_bedrock_with_retry(messages, system_prompts)
                ai_response = parse_converse_response(response)
                extracted_info, mapping = parse_extraction_response(ai_response, field_names)
                result = finalize_extraction_result(extracted_info, mapping)
            else:
                logger.info("OCR無効: without_ocrモードで単一画像情報抽出を実行")
                messages, system_prompts = build_single_image_without_ocr_request(
                    image_bytes=image_bytes,
                    app_extraction_fields=app_extraction_fields,
                    field_names=field_names,
                    custom_prompt=custom_prompt
                )
                response = call_bedrock(messages, system_prompts)
                ai_response = parse_converse_response(response)
                extracted_info = extract_json_from_response(ai_response)
                if not extracted_info:
                    extracted_info = {"error": "Failed to extract JSON from response"}
                result = finalize_extraction_result(extracted_info)
                result["mapping"] = {}

            update_extracted_info(
                self.image_id,
                result["extracted_info"],
                result.get("mapping", {}),
                'completed'
            )
            update_image_status(self.image_id, "completed")

            logger.info(f"単一画像情報抽出完了: {self.image_id}")

        except Exception as e:
            logger.error(f"単一画像情報抽出エラー: {str(e)}")
            update_image_status(self.image_id, "failed")
            raise


# ===== サービスクラス =====

class ExtractionService:
    """情報抽出処理を管理するサービスクラス"""

    def __init__(self, background_task: Optional[BackgroundTaskExtension] = None):
        self.background_task = background_task

    async def get_extraction_result(self, image_id: str) -> Dict[str, Any]:
        """情報抽出結果を取得する"""
        try:
            image_data = get_image(image_id)

            if not image_data:
                logger.warning(f"画像が見つかりません (image_id: {image_id})")
                raise ValueError("画像が見つかりません")

            app_name = image_data.get("app_name")
            if not app_name:
                logger.error(f"app_name not found for image {image_id}")
                raise ValueError(f"app_name not found for image {image_id}")
            
            app_display_name = get_app_display_name(app_name)
            app_extraction_fields = get_extraction_fields_for_app(app_name)[
                "fields"]

            extraction_status = image_data.get("extraction_status")
            if extraction_status != "completed":
                logger.info(f"抽出処理が完了していません (status: {extraction_status})")
                return {
                    "extracted_info": {},
                    "mapping": {},
                    "status": extraction_status or "not_started",
                    "app_name": app_name,
                    "app_display_name": app_display_name,
                    "fields": app_extraction_fields,
                    "verification_completed": image_data.get("verification_completed", False),
                    "verification_completed_at": image_data.get("verification_completed_at")
                }

            extracted_info = image_data.get("extracted_info", {})
            extraction_mapping = image_data.get("extraction_mapping", {})

            logger.info(
                f"DBから取得した抽出情報 (型: {type(extracted_info)}): {extracted_info}")
            logger.info(
                f"DBから取得したマッピング (型: {type(extraction_mapping)}): {extraction_mapping}")

            extracted_info = decimal_to_float(extracted_info)
            extraction_mapping = decimal_to_float(extraction_mapping)

            result = {
                "extracted_info": extracted_info,
                "mapping": extraction_mapping,
                "status": extraction_status,
                "app_name": app_name,
                "app_display_name": app_display_name,
                "fields": app_extraction_fields,
                "verification_completed": image_data.get("verification_completed", False),
                "verification_completed_at": image_data.get("verification_completed_at")
            }

            logger.info(f"Retrieved extraction result for image {image_id}")
            return result

        except Exception as e:
            logger.error(f"Error getting extraction result: {str(e)}")
            raise

    async def start_extraction(self, image_id: str, request: ExtractionRequest) -> Dict[str, Any]:
        """情報抽出を開始する"""
        try:
            logger.info(f"情報抽出を開始: {image_id}")

            self.extract_information(image_id)

            # 結果を取得
            image_data = get_image(image_id)
            extracted_info = image_data.get("extracted_info", {})

            logger.info(f"情報抽出完了: {image_id}")
            return {"status": "success", "extracted_info": extracted_info}

        except Exception as e:
            logger.error(f"情報抽出エラー: {str(e)}")
            update_image_status(image_id, "failed")
            raise

    async def get_extraction_status(self, image_id: str) -> Dict[str, Any]:
        """情報抽出のステータスを取得する"""
        try:
            image_data = get_image(image_id)

            if not image_data:
                raise ValueError("Image not found")

            return {"status": image_data.get("extraction_status") or "not_started"}
        except Exception as e:
            logger.error(f"Error getting extraction status: {str(e)}")
            raise

    async def update_extraction_result(self, image_id: str, edited_data: dict) -> None:
        """情報抽出結果を更新する"""
        try:
            extracted_info = edited_data.get("extracted_info", {})
            mapping = edited_data.get("mapping", {})

            update_extracted_info(image_id, extracted_info, mapping)

            logger.info(f"Updated extraction result for image {image_id}")

        except Exception as e:
            logger.error(f"Error updating extraction result: {str(e)}")
            raise

    def extract_information(self, image_id: str) -> None:
        """OCR結果から情報抽出を実行"""
        try:
            logger.info(
                f"Starting information extraction for image {image_id}")

            image_data = get_image(image_id)
            if not image_data:
                raise ValueError(f"Image not found: {image_id}")

            extractor = self._get_extractor(image_id, image_data)
            extractor.extract()
            sync_parent_status(image_id)

            logger.info(
                f"Successfully completed extraction for image {image_id}")

        except Exception as e:
            logger.error(f"Error during information extraction: {str(e)}")
            sync_parent_status(image_id)
            raise

    def _get_extractor(self, image_id: str, image_data: dict):
        """処理モードに応じた抽出器を返す"""
        page_processing_mode = image_data.get(
            "page_processing_mode", "combined")
        converted_s3_keys = image_data.get("converted_s3_key")

        is_multiimage_combined = (
            page_processing_mode == "combined" and
            isinstance(converted_s3_keys, list) and
            len(converted_s3_keys) > 1
        )

        if is_multiimage_combined:
            return MultiImageExtractor(image_id, image_data)
        else:
            return SingleImageExtractor(image_id, image_data)

    async def update_verification_status(self, image_id: str, verification_completed: bool, verified_by: str = None) -> Dict[str, Any]:
        """確認完了ステータスを更新する"""
        try:
            update_verification_status(image_id, verification_completed, verified_by=verified_by)
            return {
                "status": "success",
                "verification_completed": verification_completed
            }
        except Exception as e:
            logger.error(f"Error updating verification status: {str(e)}")
            raise
