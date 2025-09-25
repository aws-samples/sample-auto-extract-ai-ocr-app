from fastapi import APIRouter, HTTPException
import logging

from schemas import ExtractionRequest
from services.extraction_service import ExtractionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ocr/extract", tags=["Extraction"])

# 情報抽出サービスのインスタンス（main.pyでbackground_taskが設定される）
extraction_service = ExtractionService()


def set_background_task(background_task):
    """main.pyからバックグラウンドタスクを設定する"""
    global extraction_service
    extraction_service = ExtractionService(background_task)


@router.get("/{image_id}")
async def get_extraction_result(image_id: str):
    """情報抽出結果を取得する"""
    try:
        result = await extraction_service.get_extraction_result(image_id)
        return result
    except Exception as e:
        logger.error(f"Error getting extraction result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/{image_id}")
async def start_extraction(image_id: str, request: ExtractionRequest):
    """情報抽出を開始する"""
    try:
        result = await extraction_service.start_extraction(image_id, request)
        return result
    except Exception as e:
        logger.error(f"Error starting extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/status/{image_id}")
async def get_extraction_status(image_id: str):
    """情報抽出のステータスを取得する"""
    try:
        result = await extraction_service.get_extraction_status(image_id)
        return result
    except Exception as e:
        logger.error(f"Error getting extraction status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/edit/{image_id}")
async def update_extraction_result(image_id: str, edited_data: dict):
    """情報抽出結果を更新する"""
    try:
        await extraction_service.update_extraction_result(image_id, edited_data)
        return {"status": "success", "message": "Extraction results updated successfully"}
    except Exception as e:
        logger.error(f"Error updating extraction result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
