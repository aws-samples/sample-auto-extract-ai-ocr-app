import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from database import (
    get_image, update_extracted_info,
    update_image_status
)
from schemas import ExtractionRequest
from config import settings
from app_schema import (
    get_app_display_name, get_extraction_fields_for_app
)
from background import BackgroundTaskExtension
from utils import decimal_to_float
from extraction import (
    extract_information_from_multi_images_with_ocr,
    extract_information_from_single_image_with_ocr
)

logger = logging.getLogger(__name__)

DEFAULT_APP = "default"


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
        extract_information_from_multi_images_with_ocr(self.image_id)


class SingleImageExtractor(InformationExtractor):
    """単一画像情報抽出プロセッサー"""

    def extract(self) -> None:
        """従来の情報抽出（individual/単一画像共通）（統一版）"""
        logger.info(f"単一画像での情報抽出を実行: {self.image_id}")
        extract_information_from_single_image_with_ocr(self.image_id)


# ===== サービスクラス =====


class ExtractionService:
    """情報抽出処理を管理するサービスクラス"""

    def __init__(self, background_task: Optional[BackgroundTaskExtension] = None):
        self.background_task = background_task

    async def get_extraction_result(self, image_id: str) -> Dict[str, Any]:
        """情報抽出結果を取得する"""
        try:
            # 画像情報を取得
            image_data = get_image(image_id)

            if not image_data:
                logger.warning(f"画像が見つかりません (image_id: {image_id})")
                raise ValueError("画像が見つかりません")

            # アプリ名を取得（なければデフォルト）
            app_name = image_data.get("app_name", DEFAULT_APP)

            # アプリの表示名を取得
            app_display_name = get_app_display_name(app_name)

            # このアプリ用の抽出フィールド定義を取得
            app_extraction_fields = get_extraction_fields_for_app(app_name)[
                "fields"]

            # 抽出処理が完了していない場合
            extraction_status = image_data.get("extraction_status")
            if extraction_status != "completed":
                logger.info(f"抽出処理が完了していません (status: {extraction_status})")
                return {
                    "extracted_info": {},
                    "mapping": {},
                    "status": extraction_status or "not_started",
                    "app_name": app_name,
                    "app_display_name": app_display_name,
                    "fields": app_extraction_fields
                }

            extracted_info = image_data.get("extracted_info", {})
            extraction_mapping = image_data.get("extraction_mapping", {})

            logger.info(
                f"DBから取得した抽出情報 (型: {type(extracted_info)}): {extracted_info}")
            logger.info(
                f"DBから取得したマッピング (型: {type(extraction_mapping)}): {extraction_mapping}")

            # Decimal型をfloat型に変換
            extracted_info = decimal_to_float(extracted_info)
            extraction_mapping = decimal_to_float(extraction_mapping)

            result = {
                "extracted_info": extracted_info,
                "mapping": extraction_mapping,
                "status": extraction_status,
                "app_name": app_name,
                "app_display_name": app_display_name,
                "fields": app_extraction_fields
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

            # 画像データを取得
            image_data = get_image(image_id)
            if not image_data:
                raise ValueError("Image not found")

            # ページ処理モードを確認
            page_processing_mode = image_data.get(
                "page_processing_mode", "combined")
            converted_s3_keys = image_data.get("converted_s3_key")

            # 複数画像かどうかを判定
            is_multiimage = (
                page_processing_mode == "combined" and
                isinstance(converted_s3_keys, list) and
                len(converted_s3_keys) > 1
            )

            if is_multiimage:
                # 複数画像での情報抽出
                logger.info(f"複数画像情報抽出を実行: {len(converted_s3_keys)}ページ")
                return await self._extract_information_multiimage(image_id, request.dict())
            else:
                # 従来の単一画像での情報抽出
                logger.info("単一画像情報抽出を実行")
                return await self._extract_information_single(image_id, request.dict())

        except Exception as e:
            logger.error(f"情報抽出エラー: {str(e)}")
            update_image_status(image_id, "failed")
            raise

    async def _extract_information_multiimage(self, image_id: str, extraction_data: dict) -> Dict[str, Any]:
        """複数画像での情報抽出処理"""
        try:
            # 状態を更新
            update_image_status(image_id, "processing")

            # extraction.pyの関数を直接呼び出し
            extract_information_from_multi_images_with_ocr(image_id)

            # 結果を取得
            image_data = get_image(image_id)
            extracted_info = image_data.get("extracted_info", {})

            logger.info(f"複数画像情報抽出完了: {image_id}")
            return {"status": "success", "extracted_info": extracted_info}

        except Exception as e:
            logger.error(f"複数画像情報抽出エラー: {str(e)}")
            update_image_status(image_id, "failed")
            raise

    async def _extract_information_single(self, image_id: str, extraction_data: dict) -> Dict[str, Any]:
        """単一画像での情報抽出処理"""
        try:
            # 状態を更新
            update_image_status(image_id, "processing")

            # OCR結果を取得
            image_data = get_image(image_id)
            ocr_result = image_data.get("ocr_result", {})
            ocr_text = ocr_result.get("text", "")

            # extraction.pyの関数を直接呼び出し（統一版）
            extract_information_from_single_image_with_ocr(image_id)

            # 結果を取得
            updated_image_data = get_image(image_id)
            extracted_info = updated_image_data.get("extracted_info", {})

            logger.info(f"単一画像情報抽出完了: {image_id}")
            return {"status": "success", "extracted_info": extracted_info}

        except Exception as e:
            logger.error(f"単一画像情報抽出エラー: {str(e)}")
            update_image_status(image_id, "failed")
            raise

    async def get_extraction_status(self, image_id: str) -> Dict[str, Any]:
        """情報抽出のステータスを取得する"""
        try:
            # 画像情報を取得
            image_data = get_image(image_id)

            if not image_data:
                raise ValueError("Image not found")

            return {"status": image_data.get("extraction_status") or "not_started"}
        except Exception as e:
            logger.error(f"Error getting extraction status: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Error getting extraction status: {str(e)}")
            raise

    async def update_extraction_result(self, image_id: str, edited_data: dict) -> None:
        """情報抽出結果を更新する"""
        try:
            # 抽出情報を更新
            extracted_info = edited_data.get("extracted_info", {})
            mapping = edited_data.get("mapping", {})

            update_extracted_info(image_id, extracted_info, mapping)

            logger.info(f"Updated extraction result for image {image_id}")

        except Exception as e:
            logger.error(f"Error updating extraction result: {str(e)}")
            raise

    def extract_information(self, image_id: str) -> None:
        """OCR結果から情報抽出のみを実行（新設計）"""
        try:
            logger.info(
                f"Starting information extraction for image {image_id}")

            # 画像データを取得（処理を実行）
            image_data = get_image(image_id)
            if not image_data:
                raise ValueError(f"Image not found: {image_id}")

            # 抽出モード判定
            extractor = self._get_extractor(image_id, image_data)

            # 情報抽出実行
            extractor.extract()

            logger.info(
                f"Successfully completed extraction for image {image_id}")

        except Exception as e:
            logger.error(f"Error during information extraction: {str(e)}")
            # エラー時は状態更新しない（各関数内で処理済み）
            raise

    def _get_extractor(self, image_id: str, image_data: dict):
        """処理モードに応じた抽出器を返す（画像の種類と処理モードに応じて判定）"""
        # クラスは同じファイル内に定義済み

        # 複数画像（combinedモード）かどうかを判定（処理を実行）
        page_processing_mode = image_data.get(
            "page_processing_mode", "combined")
        converted_s3_keys = image_data.get("converted_s3_key")

        is_multiimage_combined = (
            page_processing_mode == "combined" and
            isinstance(converted_s3_keys, list) and
            len(converted_s3_keys) > 1
        )

        # 抽出器選択（処理モードに応じて選択）
        if is_multiimage_combined:
            return MultiImageExtractor(image_id, image_data)
        else:
            return SingleImageExtractor(image_id, image_data)
