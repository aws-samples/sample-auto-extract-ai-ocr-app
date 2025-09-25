import uuid
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from database import (
    create_job, get_images, get_job, get_images_by_job_id,
    get_image, update_ocr_result as db_update_ocr_result,
    update_image_status
)
from schemas import OcrResult, OcrResultResponse
from config import settings
from background import BackgroundTaskExtension
from ocr import perform_ocr_multipage, perform_ocr_individual_page, perform_ocr_single_image

logger = logging.getLogger(__name__)


class OcrProcessor(ABC):
    """OCR処理の基底クラス"""

    def __init__(self, image_id: str):
        self.image_id = image_id

    @abstractmethod
    def execute_ocr(self) -> None:
        """OCR処理を実行"""
        pass


class MultipageOcrProcessor(OcrProcessor):
    """複数画像統合処理プロセッサー"""

    def execute_ocr(self) -> None:
        """複数ページのPDFを統合してOCR処理を実行"""
        logger.info(f"複数画像統合処理を実行: {self.image_id}")
        perform_ocr_multipage(self.image_id)


class IndividualPageOcrProcessor(OcrProcessor):
    """個別ページ処理プロセッサー"""

    def execute_ocr(self) -> None:
        """PDFから分割された個別ページのOCR処理を実行"""
        logger.info(f"個別ページ処理を実行: {self.image_id}")
        perform_ocr_individual_page(self.image_id)


class SingleImageOcrProcessor(OcrProcessor):
    """単一画像処理プロセッサー"""

    def execute_ocr(self) -> None:
        """単一画像ファイルのOCR処理を実行"""
        logger.info(f"単一画像処理を実行: {self.image_id}")
        perform_ocr_single_image(self.image_id)


class OcrService:
    """OCR処理を管理するサービスクラス"""

    def __init__(self, background_task: Optional[BackgroundTaskExtension] = None):
        self.enable_ocr = settings.ENABLE_OCR
        self.background_task = background_task

    async def start_ocr_job(self, app_name: Optional[str] = None) -> str:
        """OCR処理ジョブを開始する"""
        job_id = str(uuid.uuid4())

        try:
            # ジョブを作成
            create_job(job_id, 'processing')

            # 保留中の画像を取得（app_name指定時はGSIでquery、未指定時はscanで全件取得）
            if app_name:
                # 特定アプリの画像のみをGSI経由で効率的に取得（DynamoDB scanの1MB制限を回避）
                images_list = get_images(app_name)
                logger.info(
                    f"アプリ '{app_name}' の画像を取得しました: {len(images_list)}件")
            else:
                # 全アプリの画像を取得（小規模データ用、大量データがある場合は要注意）
                images_list = get_images()
                logger.warning(
                    "全アプリの画像をscanで取得中（大量データがある場合は処理が不完全になる可能性があります）")

            processing_images = []

            # pendingステータスの画像のみを処理対象とする
            for image in images_list:
                if image.get("status") == "pending":
                    update_image_status(image.get("id"), "processing", job_id)
                    processing_images.append(image)

            # バックグラウンドタスクとしてOCR処理を実行
            if processing_images:
                logger.info(
                    f"バックグラウンドタスクを開始します: job_id={job_id}, images={len(processing_images)}")
                if self.background_task:
                    # バックグラウンドタスクとして実行
                    task_id = self.background_task.add_task(
                        self._process_job_pipeline, job_id)
                    logger.info(
                        f"Started OCR job {job_id} with task ID {task_id}")
                else:
                    # 同期実行（テスト用）
                    await self._process_ocr_background(job_id, processing_images, app_name)
            else:
                logger.warning(f"処理対象の画像がありません: job_id={job_id}")

            logger.info(f"Started OCR job: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"OCRジョブの開始エラー: {str(e)}")
            raise

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """ジョブステータスを取得する"""
        try:
            return get_job(job_id)
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            raise

    async def get_ocr_result(self, image_id: str) -> OcrResultResponse:
        """OCR結果を取得する"""
        image_data = get_image(image_id)

        if not image_data:
            raise ValueError("Image not found")

        ocr_result = image_data.get("ocr_result", {})

        # OCR無効時はocr_resultが存在しない
        if ocr_result is None:
            ocr_result = {}

        # 画像URLを生成
        image_url = f"{settings.API_BASE_URL}/image/{image_id}"

        # s3_keyがリスト形式の場合は最初の要素を取得
        s3_key = image_data.get("s3_key")
        if isinstance(s3_key, list):
            s3_key = s3_key[0] if s3_key else ""

        return OcrResultResponse(
            filename=image_data.get("filename"),
            s3_key=s3_key,
            uploadTime=image_data.get("upload_time"),
            status=image_data.get("status"),
            ocrResult=OcrResult(
                **ocr_result) if ocr_result else OcrResult(words=[]),
            imageUrl=image_url
        )

    async def update_ocr_result(self, image_id: str, edited_ocr_data: dict) -> None:
        """OCR結果を更新する"""
        db_update_ocr_result(image_id, edited_ocr_data)

    def _process_job_pipeline(self, job_id: str) -> None:
        """バックグラウンドタスク用のジョブパイプライン処理"""
        try:
            logger.info(f"バックグラウンドタスク開始: job_id={job_id}")
            # ジョブに関連する画像を取得
            images = get_images_by_job_id(job_id)
            logger.info(f"Processing job {job_id} with {len(images)} images")

            # 同時処理数を制限（例: 最大2枚ずつ処理）
            batch_size = 2
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1} with {len(batch)} images")

                for image in batch:
                    image_id = image.get("id")
                    # 新実装（ImageProcessingPipelineを直接使用）
                    from services.image_processing_pipeline import ImageProcessingPipeline
                    pipeline = ImageProcessingPipeline()
                    pipeline.process_complete_pipeline(image_id)

        except Exception as e:
            logger.error(f"Error in background OCR processing: {str(e)}")
            raise

    def process_image_ocr(self, image_id: str) -> None:
        """画像のOCR処理のみを実行"""
        try:
            logger.info(f"Processing single image: {image_id}")

            # 画像情報を取得
            image_data = get_image(image_id)
            if not image_data:
                raise ValueError(f"Image not found: {image_id}")

            # ステータスを処理中に更新
            update_image_status(image_id, "processing")

            # 処理モードを判定してOCRプロセッサーを選択
            processor = self._get_ocr_processor(image_id, image_data)

            # OCR実行
            processor.execute_ocr()

            logger.info(f"Successfully completed OCR for image {image_id}")

        except Exception as e:
            logger.error(
                f"Error processing OCR for image {image_id}: {str(e)}")
            update_image_status(image_id, "failed")
            raise

    def _get_ocr_processor(self, image_id: str, image_data: dict):
        """画像の種類と処理モードに応じて適切なOCRプロセッサーを返す"""
        # クラスは同じファイル内に定義済み

        # ページ処理モードを確認（デフォルトはcombined）
        page_processing_mode = image_data.get(
            "page_processing_mode", "combined")
        converted_s3_keys = image_data.get("converted_s3_key")

        # 複数画像をcombinedモードで処理するかを判定
        is_multiimage_combined = (
            page_processing_mode == "combined" and
            isinstance(converted_s3_keys, list) and
            len(converted_s3_keys) > 1
        )

        # PDFから分割された個別ページかを判定
        is_individual_page = image_data.get("parent_document_id") is not None

        logger.info(
            f"Processing image {image_id} (mode: {page_processing_mode})")

        # 処理モードに応じてプロセッサーを選択
        if is_multiimage_combined:
            return MultipageOcrProcessor(image_id)
        elif is_individual_page:
            return IndividualPageOcrProcessor(image_id)
        else:
            return SingleImageOcrProcessor(image_id)
