"""
画像処理パイプライン
OCR→情報抽出の完全フローを管理
アップロードされた画像に対してOCR処理と情報抽出を順次実行
"""
import logging
from services.ocr_service import OcrService
from services.extraction_service import ExtractionService

logger = logging.getLogger(__name__)


class ImageProcessingPipeline:
    """OCR→情報抽出の完全パイプライン"""

    def __init__(self):
        self.ocr_service = OcrService()
        self.extraction_service = ExtractionService()

    def process_complete_pipeline(self, image_id: str) -> None:
        """OCR→情報抽出の完全パイプラインを実行"""
        try:
            logger.info(f"Starting complete pipeline for image {image_id}")

            # 1. OCR処理
            self.ocr_service.process_image_ocr(image_id)

            # 2. 情報抽出処理
            self.extraction_service.extract_information(image_id)

            logger.info(
                f"Successfully completed pipeline for image {image_id}")

        except Exception as e:
            logger.error(f"Pipeline failed for {image_id}: {e}")
            # 既存ロジックに合わせて、エラーハンドリングは各サービス内で実行済み
            raise
