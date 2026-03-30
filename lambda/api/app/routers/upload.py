from fastapi import APIRouter, HTTPException, Request, Depends
import logging

from schemas import (
    PresignedUrlRequest, PresignedUrlResponse, UploadCompleteRequest,
)
from services.upload_service import UploadService
from dependencies.services import get_upload_service
from utils.auth import require_auth, get_cognito_sub, RequirePermission

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Upload"])


@router.post("/generate-presigned-url", response_model=PresignedUrlResponse)
async def generate_presigned_url(
    request: PresignedUrlRequest,
    req: Request,
    user=Depends(RequirePermission("viewer")),
    service: UploadService = Depends(get_upload_service),
):
    """署名付きURLを生成して返す（対象ユースケースに viewer 以上の権限が必要）"""
    try:
        sub = get_cognito_sub(req)
        return await service.generate_presigned_url(request, uploaded_by=sub)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/upload-complete")
async def upload_complete(request: UploadCompleteRequest, service: UploadService = Depends(get_upload_service)):
    """アップロード完了を処理する"""
    try:
        return await service.handle_upload_complete(request)
    except Exception as e:
        logger.error(f"Error handling upload complete: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/image/{image_id}")
async def get_image_stream(image_id: str, service: UploadService = Depends(get_upload_service)):
    """画像を取得して返す"""
    try:
        return await service.get_image_stream(image_id)
    except Exception as e:
        logger.error(f"Error getting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/generate-presigned-download-url/{image_id}")
async def generate_presigned_download_url(image_id: str, service: UploadService = Depends(get_upload_service)):
    """ダウンロード用の署名付きURLを生成する"""
    try:
        return await service.generate_download_url(image_id)
    except Exception as e:
        logger.error(f"Error generating download URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/images")
async def get_images(
    app_name: str = None,
    req: Request = None,
    user=Depends(require_auth),
    service: UploadService = Depends(get_upload_service),
):
    """画像一覧を取得する（権限のあるユースケースの画像をすべて返す）"""
    try:
        return await service.get_images_for_user(
            user_id=str(user["id"]),
            role=user["role"],
            app_name=app_name,
        )
    except Exception as e:
        logger.error(f"Error getting images list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/images/{image_id}")
async def delete_image(
    image_id: str,
    req: Request,
    user=Depends(require_auth),
    service: UploadService = Depends(get_upload_service),
):
    """画像を削除する（所有者 or admin のみ）"""
    try:
        sub = get_cognito_sub(req)
        return await service.delete_image(
            image_id,
            cognito_sub=sub,
            is_admin=(user["role"] == "admin"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
