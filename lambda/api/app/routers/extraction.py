from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import logging

from schemas import ExtractionRequest
from services.extraction_service import ExtractionService
from dependencies.services import get_extraction_service
from dependencies.auth import get_cognito_sub, RequireImagePermission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ocr/extract", tags=["Extraction"])


@router.get("/{image_id}")
async def get_extraction_result(image_id: str, user=Depends(RequireImagePermission("viewer")), service: ExtractionService = Depends(get_extraction_service)):
    """情報抽出結果を取得する（対象画像に viewer 以上の権限が必要）"""
    try:
        return await service.get_extraction_result(image_id)
    except Exception as e:
        logger.error(f"Error getting extraction result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/{image_id}")
async def start_extraction(image_id: str, request: ExtractionRequest, user=Depends(RequireImagePermission("viewer")), service: ExtractionService = Depends(get_extraction_service)):
    """情報抽出を開始する（対象画像に viewer 以上の権限が必要）"""
    try:
        return await service.start_extraction(image_id, request)
    except Exception as e:
        logger.error(f"Error starting extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/status/{image_id}")
async def get_extraction_status(image_id: str, user=Depends(RequireImagePermission("viewer")), service: ExtractionService = Depends(get_extraction_service)):
    """情報抽出のステータスを取得する（対象画像に viewer 以上の権限が必要）"""
    try:
        return await service.get_extraction_status(image_id)
    except Exception as e:
        logger.error(f"Error getting extraction status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/edit/{image_id}")
async def update_extraction_result(image_id: str, edited_data: dict, user=Depends(RequireImagePermission("viewer")), service: ExtractionService = Depends(get_extraction_service)):
    """情報抽出結果を更新する（対象画像に viewer 以上の権限が必要）"""
    try:
        await service.update_extraction_result(image_id, edited_data)
        return {"status": "success", "message": "Extraction results updated successfully"}
    except Exception as e:
        logger.error(f"Error updating extraction result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


class VerificationRequest(BaseModel):
    verification_completed: bool = False


@router.post("/verification/{image_id}")
async def update_verification_status(image_id: str, body: VerificationRequest, req: Request = None, user=Depends(RequireImagePermission("viewer")), service: ExtractionService = Depends(get_extraction_service)):
    """確認完了ステータスを更新する（対象画像に viewer 以上の権限が必要）"""
    try:
        verified_by = get_cognito_sub(req) if req else None
        return await service.update_verification_status(image_id, body.verification_completed, verified_by=verified_by)
    except Exception as e:
        logger.error(f"Error updating verification status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
