from fastapi import APIRouter, HTTPException, Depends, Request
import logging

from schemas import ExtractionRequest
from services.extraction_service import ExtractionService
from dependencies.services import get_extraction_service
from utils.auth import require_auth, get_cognito_sub

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ocr/extract", tags=["Extraction"])


@router.get("/{image_id}")
async def get_extraction_result(image_id: str, service: ExtractionService = Depends(get_extraction_service)):
    """情報抽出結果を取得する"""
    try:
        return await service.get_extraction_result(image_id)
    except Exception as e:
        logger.error(f"Error getting extraction result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/{image_id}")
async def start_extraction(image_id: str, request: ExtractionRequest, service: ExtractionService = Depends(get_extraction_service)):
    """情報抽出を開始する"""
    try:
        return await service.start_extraction(image_id, request)
    except Exception as e:
        logger.error(f"Error starting extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/status/{image_id}")
async def get_extraction_status(image_id: str, service: ExtractionService = Depends(get_extraction_service)):
    """情報抽出のステータスを取得する"""
    try:
        return await service.get_extraction_status(image_id)
    except Exception as e:
        logger.error(f"Error getting extraction status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/edit/{image_id}")
async def update_extraction_result(image_id: str, edited_data: dict, service: ExtractionService = Depends(get_extraction_service)):
    """情報抽出結果を更新する"""
    try:
        await service.update_extraction_result(image_id, edited_data)
        return {"status": "success", "message": "Extraction results updated successfully"}
    except Exception as e:
        logger.error(f"Error updating extraction result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/verification/{image_id}")
async def update_verification_status(image_id: str, request: dict, req: Request = None, user=Depends(require_auth), service: ExtractionService = Depends(get_extraction_service)):
    """確認完了ステータスを更新する"""
    try:
        verification_completed = request.get("verification_completed", False)
        verified_by = get_cognito_sub(req) if req else None
        return await service.update_verification_status(image_id, verification_completed, verified_by=verified_by)
    except Exception as e:
        logger.error(f"Error updating verification status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
