from fastapi import APIRouter, HTTPException, UploadFile, File
import logging

from schemas import (
    OcrResultResponse, OcrStartRequest, JobStartResponse, OcrResult
)
from services.ocr_service import OcrService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ocr", tags=["OCR"])

# OCRサービスのインスタンス（main.pyでbackground_taskが設定される）
ocr_service = OcrService()


def set_background_task(background_task):
    """main.pyからバックグラウンドタスクを設定する"""
    global ocr_service
    ocr_service = OcrService(background_task)


@router.post("/start", response_model=JobStartResponse)
async def start_ocr(request: OcrStartRequest = OcrStartRequest()):
    """OCR処理を開始する"""
    try:
        job_id = await ocr_service.start_ocr_job(request.app_name)
        return JobStartResponse(jobId=job_id)
    except Exception as e:
        logger.error(f"OCR job start error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/status/{job_id}")
async def get_ocr_status(job_id: str):
    """OCRジョブのステータスを取得する"""
    try:
        status = await ocr_service.get_job_status(job_id)
        return status
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/result/{image_id}", response_model=OcrResultResponse)
async def get_ocr_result(image_id: str):
    """OCR結果を取得する"""
    try:
        result = await ocr_service.get_ocr_result(image_id)
        return result
    except Exception as e:
        logger.error(f"Error getting OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/edit/{image_id}")
async def update_ocr_result(image_id: str, edited_ocr_data: dict):
    """OCR結果を更新する"""
    try:
        await ocr_service.update_ocr_result(image_id, edited_ocr_data)
        return {"status": "success", "message": "OCR results updated successfully"}
    except Exception as e:
        logger.error(f"Error updating OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
