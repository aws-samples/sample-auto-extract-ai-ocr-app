"""Agent API router."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

from services.agent_service import AgentService
from dependencies.services import get_agent_service
from dependencies.auth import RequireImagePermission, require_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ocr/agent", tags=["Agent"])


class SuggestionStatusUpdate(BaseModel):
    status: str  # "accepted" or "rejected"


@router.get("/tools")
async def get_tools(image_id: str = None, user=Depends(require_auth), service: AgentService = Depends(get_agent_service)):
    """Get tools for an image's usecase (or all tools if no image_id)"""
    try:
        if image_id:
            return await service.get_usecase_tools_for_image(image_id)
        return await service.get_available_tools()
    except Exception as e:
        logger.error(f"Error getting tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/image/{image_id}")
async def get_agent_job_by_image(image_id: str, user=Depends(RequireImagePermission("viewer"))):
    """画像の最新エージェントジョブを取得

    image レコードの agent_status を権威ソースとし、
    job が古い completed のままでも image 側が processing なら processing を返す。
    """
    try:
        from repositories.job_repository import get_latest_agent_job_by_image_id
        from repositories import get_image
        image = get_image(image_id)
        image_agent_status = image.get("agent_status") if image else None

        job = get_latest_agent_job_by_image_id(image_id)
        if not job:
            # image 側で processing になっていれば、ジョブ作成待ちの processing として返す
            if image_agent_status == "processing":
                return {"status": "processing", "suggestions": []}
            return {"status": "none", "suggestions": []}

        # image 側が processing で job がまだ古い completed の場合、
        # 新しい AgentKick がジョブ作成中とみなして processing を返す
        job_status = job.get("status")
        if image_agent_status == "processing" and job_status not in ("processing",):
            return {"status": "processing", "suggestions": []}

        suggestions = job.get("suggestions", [])
        # Only return pending suggestions with their original index
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


@router.post("/{image_id}")
async def start_agent_correction(image_id: str, user=Depends(RequireImagePermission("viewer")), service: AgentService = Depends(get_agent_service)):
    """Start agent correction job"""
    try:
        job_id = await service.start_agent_correction(image_id)
        return {"jobId": job_id}
    except Exception as e:
        logger.error(f"Error starting agent correction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/status/{job_id}")
async def get_agent_job_status(job_id: str, user=Depends(require_auth), service: AgentService = Depends(get_agent_service)):
    """Get agent correction job status"""
    try:
        return await service.get_agent_job_status(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.patch("/image/{image_id}/suggestions/{suggestion_index}")
async def update_suggestion(
    image_id: str,
    suggestion_index: int,
    body: SuggestionStatusUpdate,
    user=Depends(RequireImagePermission("viewer")),
):
    """提案の採用/却下を永続化"""
    if body.status not in ("accepted", "rejected"):
        raise HTTPException(status_code=400, detail="status must be 'accepted' or 'rejected'")
    try:
        from repositories.job_repository import update_suggestion_status
        pending_count = update_suggestion_status(image_id, suggestion_index, body.status)
        return {"ok": True, "pending_count": pending_count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating suggestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
