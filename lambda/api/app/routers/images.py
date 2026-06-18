"""Images Router - /images prefix

Upload, OCR, extraction, agent, verification endpoints unified under /images.
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
import logging
from typing import Literal, Optional

from schemas import (
    PresignedUrlRequest, PresignedUrlResponse, UploadCompleteRequest,
    OcrResultResponse,
)
from services.ocr_service import OcrService, EndpointNotReadyError
from services.upload_service import UploadService
from services.extraction_service import ExtractionService
from services.agent_service import AgentService
from dependencies.services import (
    get_ocr_service, get_upload_service, get_extraction_service, get_agent_service,
)
from dependencies.auth import (
    require_user, get_cognito_sub, check_usecase_permission,
    RequirePermission, RequireImagePermission,
)
from repositories import get_image
from repositories.job_repository import get_latest_agent_job_by_image_id, update_suggestion_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["Images"])


# === Upload ===

@router.post("/upload-url", response_model=PresignedUrlResponse)
async def generate_presigned_url(
    request: PresignedUrlRequest,
    req: Request,
    user=Depends(require_user),
    service: UploadService = Depends(get_upload_service),
):
    """署名付きURLを生成して返す"""
    try:
        check_usecase_permission(user, request.app_name, "viewer")
        sub = get_cognito_sub(req)
        return await service.generate_presigned_url(request, uploaded_by=sub)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("")
async def list_images(
    app_name: Optional[str] = None,
    user=Depends(require_user),
    service: UploadService = Depends(get_upload_service),
):
    """画像一覧を取得する"""
    try:
        return await service.get_images_for_user(
            user_id=str(user["id"]),
            role=user["role"],
            app_name=app_name,
        )
    except Exception as e:
        logger.error(f"Error getting images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    req: Request,
    user=Depends(RequireImagePermission("viewer")),
    service: UploadService = Depends(get_upload_service),
):
    """画像を削除する"""
    try:
        sub = get_cognito_sub(req)
        is_admin = user.get("role") == "admin"
        return await service.delete_image(image_id, sub, is_admin)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/{image_id}/download-url")
async def generate_presigned_download_url(
    image_id: str,
    user=Depends(RequireImagePermission("viewer")),
    service: UploadService = Depends(get_upload_service),
):
    """ダウンロード用の署名付きURLを生成する"""
    try:
        return await service.generate_download_url(image_id)
    except Exception as e:
        logger.error(f"Error generating download URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/{image_id}/upload-complete")
async def upload_complete(
    image_id: str,
    request: UploadCompleteRequest,
    user=Depends(RequireImagePermission("viewer")),
    service: UploadService = Depends(get_upload_service),
):
    """アップロード完了を処理する"""
    try:
        return await service.handle_upload_complete(image_id, request)
    except Exception as e:
        logger.error(f"Error handling upload complete: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# === Process (Pipeline) ===

class ProcessRequest(BaseModel):
    start_from: Optional[str] = None


@router.post("/{image_id}/process")
async def process_image(
    image_id: str,
    body: ProcessRequest,
    user=Depends(RequireImagePermission("viewer")),
    service: OcrService = Depends(get_ocr_service),
):
    """パイプライン実行（OCR→抽出→Agent）。body.start_from="extraction" で抽出以降のみ"""
    skip_ocr = body.start_from == "extraction"
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
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# === Status ===

@router.get("/{image_id}/status")
async def get_image_status(
    image_id: str,
    user=Depends(RequireImagePermission("viewer")),
):
    """全フェーズのステータスを一括取得（ポーリング用）"""
    try:
        image_data = get_image(image_id)
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        return {
            "extraction_status": image_data.get("extraction_status") or "not_started",
            "agent_status": image_data.get("agent_status") or "idle",
            "agent_pending_suggestions_count": image_data.get("agent_suggestions_count", 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# === OCR ===

@router.get("/{image_id}/ocr", response_model=OcrResultResponse)
async def get_ocr_result(
    image_id: str,
    user=Depends(RequireImagePermission("viewer")),
    service: OcrService = Depends(get_ocr_service),
):
    """OCR結果を取得する"""
    try:
        return await service.get_ocr_result(image_id)
    except Exception as e:
        logger.error(f"Error getting OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.put("/{image_id}/ocr")
async def update_ocr_result(
    image_id: str,
    edited_ocr_data: dict,
    user=Depends(RequireImagePermission("viewer")),
    service: OcrService = Depends(get_ocr_service),
):
    """OCR結果を更新する"""
    try:
        await service.update_ocr_result(image_id, edited_ocr_data)
        return {"status": "success", "message": "OCR results updated successfully"}
    except Exception as e:
        logger.error(f"Error updating OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# === Extraction ===

@router.get("/{image_id}/extraction")
async def get_extraction_result(
    image_id: str,
    user=Depends(RequireImagePermission("viewer")),
    service: ExtractionService = Depends(get_extraction_service),
):
    """情報抽出結果を取得する"""
    try:
        return await service.get_extraction_result(image_id)
    except Exception as e:
        logger.error(f"Error getting extraction result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.put("/{image_id}/extraction")
async def update_extraction_result(
    image_id: str,
    edited_data: dict,
    user=Depends(RequireImagePermission("viewer")),
    service: ExtractionService = Depends(get_extraction_service),
):
    """情報抽出結果を更新する"""
    try:
        await service.update_extraction_result(image_id, edited_data)
        return {"status": "success", "message": "Extraction results updated successfully"}
    except Exception as e:
        logger.error(f"Error updating extraction result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# === Verification ===

class VerificationRequest(BaseModel):
    verification_completed: bool = False


@router.patch("/{image_id}/verification")
async def update_verification_status(
    image_id: str,
    body: VerificationRequest,
    req: Request,
    user=Depends(RequireImagePermission("viewer")),
    service: ExtractionService = Depends(get_extraction_service),
):
    """確認完了ステータスを更新する"""
    try:
        verified_by = get_cognito_sub(req)
        return await service.update_verification_status(image_id, body.verification_completed, verified_by=verified_by)
    except Exception as e:
        logger.error(f"Error updating verification status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# === Agent ===

class SuggestionStatusUpdate(BaseModel):
    status: Literal["accepted", "rejected"]


@router.post("/{image_id}/agent")
async def start_agent_correction(
    image_id: str,
    user=Depends(RequireImagePermission("viewer")),
    service: AgentService = Depends(get_agent_service),
):
    """Agent検証を開始する"""
    try:
        job_id = await service.start_agent_correction(image_id)
        return {"jobId": job_id}
    except Exception as e:
        logger.error(f"Error starting agent correction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/{image_id}/agent")
async def get_agent_job_by_image(
    image_id: str,
    user=Depends(RequireImagePermission("viewer")),
):
    """画像の最新エージェントジョブを取得"""
    try:
        image = get_image(image_id)
        image_agent_status = image.get("agent_status") if image else None

        job = get_latest_agent_job_by_image_id(image_id)
        if not job:
            if image_agent_status == "processing":
                return {"status": "processing", "suggestions": []}
            return {"status": "none", "suggestions": []}

        job_status = job.get("status")
        if image_agent_status == "processing" and job_status not in ("processing",):
            return {"status": "processing", "suggestions": []}

        suggestions = job.get("suggestions", [])
        pending = []
        for i, s in enumerate(suggestions):
            if s.get("status", "pending") == "pending":
                pending.append({**s, "index": i})
        return {
            "job_id": job.get("id"),
            "status": job_status,
            "suggestions": pending,
            "total_suggestions_count": len(suggestions),
            "error": job.get("error"),
        }
    except Exception as e:
        logger.error(f"Error getting agent job by image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/{image_id}/agent/tools")
async def get_agent_tools_for_image(
    image_id: str,
    user=Depends(RequireImagePermission("viewer")),
    service: AgentService = Depends(get_agent_service),
):
    """画像のユースケースに紐づくツール一覧"""
    try:
        return await service.get_usecase_tools_for_image(image_id)
    except Exception as e:
        logger.error(f"Error getting tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.patch("/{image_id}/agent/suggestions/{suggestion_index}")
async def update_suggestion(
    image_id: str,
    suggestion_index: int,
    body: SuggestionStatusUpdate,
    user=Depends(RequireImagePermission("viewer")),
):
    """提案の採用/却下を永続化"""
    try:
        pending_count = update_suggestion_status(image_id, suggestion_index, body.status)
        return {"ok": True, "pending_count": pending_count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating suggestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
