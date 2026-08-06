"""Step Functions Lambda ハンドラー

FastAPI Lambda とは別の Lambda として実行される。
OCR → 情報抽出の完全パイプラインを画像ごとに実行する。
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.ocr_service import OcrService
from services.extraction_service import ExtractionService
from config import settings

logger = logging.getLogger(__name__)


def process_image_handler(event, context):
    """Step Functions 用: 1 枚の画像を処理

    Args:
        event: {"image_id": str, "job_id": str, "skip_ocr": bool (optional)}

    Returns:
        {"image_id": str, "success": bool, "error": str (optional)}
    """
    image_id = event["image_id"]
    skip_ocr = event.get("skip_ocr", False)

    logger.info(f"Processing image: {image_id}, skip_ocr: {skip_ocr}")

    try:
        ocr_service = OcrService()
        extraction_service = ExtractionService()

        should_skip_ocr = skip_ocr or not settings.ENABLE_OCR

        if not should_skip_ocr:
            ocr_service.process_image_ocr(image_id)

        extraction_service.extract_information(image_id)

        logger.info(f"Successfully processed image: {image_id}")
        return {"image_id": image_id, "success": True}

    except Exception as e:
        logger.error(f"Error processing {image_id}: {str(e)}")
        return {"image_id": image_id, "success": False, "error": str(e)}
