from fastapi import APIRouter, HTTPException, Depends
import logging

from schemas import (
    OcrResultResponse, OcrStartRequest, JobStartResponse
)
from services.ocr_service import OcrService, EndpointNotReadyError
from dependencies.services import get_ocr_service
from dependencies.auth import RequireImagePermission, RequirePermission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post("/start/{app_name}", response_model=JobStartResponse)
async def start_ocr(app_name: str, user=Depends(RequirePermission("viewer")), service: OcrService = Depends(get_ocr_service)):
    """OCR処理を開始する（対象ユースケースに viewer 以上の権限が必要）"""
    try:
        request = OcrStartRequest(app_name=app_name)
        result = await service.start_step_functions_job(request)
        return JobStartResponse(jobId=result["jobId"])
    except EndpointNotReadyError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "endpoint_not_ready", "message": str(e)}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting OCR job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/result/{image_id}", response_model=OcrResultResponse)
async def get_ocr_result(image_id: str, user=Depends(RequireImagePermission("viewer")), service: OcrService = Depends(get_ocr_service)):
    """OCR結果を取得する（対象画像に viewer 以上の権限が必要）"""
    try:
        return await service.get_ocr_result(image_id)
    except Exception as e:
        logger.error(f"Error getting OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/edit/{image_id}")
async def update_ocr_result(image_id: str, edited_ocr_data: dict, user=Depends(RequireImagePermission("viewer")), service: OcrService = Depends(get_ocr_service)):
    """OCR結果を更新する（対象画像に viewer 以上の権限が必要）"""
    try:
        await service.update_ocr_result(image_id, edited_ocr_data)
        return {"status": "success", "message": "OCR results updated successfully"}
    except Exception as e:
        logger.error(f"Error updating OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/start/{image_id}")
async def start_ocr_for_image(image_id: str, skip_ocr: bool = False, user=Depends(RequireImagePermission("viewer")), service: OcrService = Depends(get_ocr_service)):
    """指定した画像IDのOCR処理を開始する（対象画像に viewer 以上の権限が必要）"""
    try:
        result = await service.start_step_functions_for_image(image_id, skip_ocr)
        return result
    except EndpointNotReadyError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "endpoint_not_ready", "message": str(e)}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting OCR for image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/endpoint-status")
async def get_endpoint_status(service: OcrService = Depends(get_ocr_service)):
    """エンドポイントの状態を確認（ポーリング用）"""
    try:
        return service.get_endpoint_status()
    except Exception as e:
        logger.error(f"Error checking endpoint status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
