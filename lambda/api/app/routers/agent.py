"""Agent API router."""
from fastapi import APIRouter, HTTPException, Depends
import logging

from services.agent_service import AgentService
from dependencies.services import get_agent_service
from dependencies.auth import RequireImagePermission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ocr/agent", tags=["Agent"])


@router.get("/tools")
async def get_tools(service: AgentService = Depends(get_agent_service)):
    """Get available tools from AgentCore Runtime"""
    try:
        return await service.get_available_tools()
    except Exception as e:
        logger.error(f"Error getting tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/image/{image_id}")
async def get_agent_job_by_image(image_id: str, user=Depends(RequireImagePermission("viewer"))):
    """画像の最新エージェントジョブを取得"""
    try:
        from repositories.job_repository import get_latest_agent_job_by_image_id
        job = get_latest_agent_job_by_image_id(image_id)
        if not job:
            return {"status": "none", "suggestions": []}
        return {
            "job_id": job.get("id"),
            "status": job.get("status"),
            "suggestions": job.get("suggestions", []),
            "error": job.get("error"),
        }
    except Exception as e:
        logger.error(f"Error getting agent job by image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/{image_id}")
async def start_agent_correction(image_id: str, service: AgentService = Depends(get_agent_service)):
    """Start agent correction job"""
    try:
        job_id = await service.start_agent_correction(image_id)
        return {"jobId": job_id}
    except Exception as e:
        logger.error(f"Error starting agent correction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/status/{job_id}")
async def get_agent_job_status(job_id: str, service: AgentService = Depends(get_agent_service)):
    """Get agent correction job status"""
    try:
        return await service.get_agent_job_status(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
