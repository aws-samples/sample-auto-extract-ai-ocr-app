"""Jobs Router - /jobs prefix

Unified job status polling.
"""
from fastapi import APIRouter, HTTPException, Depends
import logging

from services.agent_service import AgentService
from dependencies.services import get_agent_service
from dependencies.auth import require_user, check_usecase_permission
from repositories.job_repository import get_job
from repositories import get_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}")
async def get_job_status(
    job_id: str,
    user=Depends(require_user),
    service: AgentService = Depends(get_agent_service),
):
    """ジョブステータスを取得する"""
    try:
        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        image = get_image(job.get("image_id", ""))
        if not image:
            raise HTTPException(status_code=404, detail="Associated image not found")
        check_usecase_permission(user, image.get("app_name", ""), "viewer")

        return await service.get_agent_job_status(job_id)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
