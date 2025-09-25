from fastapi import APIRouter, HTTPException
import logging

from schemas import (
    PresignedUrlRequest, PresignedUrlResponse, UploadCompleteRequest,
)
from services.upload_service import UploadService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Upload"])

# アップロードサービスのインスタンス
upload_service = UploadService()


@router.post("/generate-presigned-url", response_model=PresignedUrlResponse)
async def generate_presigned_url(request: PresignedUrlRequest):
    """署名付きURLを生成して返す"""
    try:
        result = await upload_service.generate_presigned_url(request)
        return result
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/upload-complete")
async def upload_complete(request: UploadCompleteRequest):
    """アップロード完了を処理する"""
    try:
        result = await upload_service.handle_upload_complete(request)
        return result
    except Exception as e:
        logger.error(f"Error handling upload complete: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """画像を取得して返す"""
    try:
        return await upload_service.get_image_stream(image_id)
    except Exception as e:
        logger.error(f"Error getting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/generate-presigned-download-url/{image_id}")
async def generate_presigned_download_url(image_id: str):
    """ダウンロード用の署名付きURLを生成する"""
    try:
        result = await upload_service.generate_download_url(image_id)
        return result
    except Exception as e:
        logger.error(f"Error generating download URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/images")
async def get_images():
    """画像一覧を取得する"""
    try:
        result = await upload_service.get_images_list()
        return result
    except Exception as e:
        logger.error(f"Error getting images list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
