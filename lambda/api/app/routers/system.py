"""System Router - /system prefix

Infrastructure status endpoints (no auth required).
"""
from fastapi import APIRouter, HTTPException, Depends
import logging

from services.ocr_service import OcrService
from dependencies.services import get_ocr_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/system", tags=["System"])


@router.get("/ocr-endpoint-status")
async def get_ocr_endpoint_status(
    service: OcrService = Depends(get_ocr_service),
):
    """OCRエンドポイントの状態を確認（ポーリング用、認証不要）"""
    try:
        return service.get_endpoint_status()
    except Exception as e:
        logger.error(f"Error checking endpoint status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
