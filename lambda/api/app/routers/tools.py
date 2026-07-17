"""Tools Router - /tools prefix

Global tool listing.
"""
from fastapi import APIRouter, Depends
import logging

from services.agent_service import AgentService
from dependencies.services import get_agent_service
from dependencies.auth import require_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["Tools"])


@router.get("")
async def get_all_tools(
    user=Depends(require_user),
    service: AgentService = Depends(get_agent_service),
):
    """全ツール一覧を取得する"""
    return await service.get_available_tools()
