"""共有設定 API"""
from fastapi import APIRouter, HTTPException, Depends
import logging

from dependencies.auth import RequirePermission
from schemas.sharing import ShareUserRequest, ShareGroupRequest
from services.sharing_service import SharingService, LastOwnerError
from dependencies.services import get_sharing_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/apps/{app_name}/sharing", tags=["Sharing"])


@router.get("")
async def get_sharing(app_name: str, user=Depends(RequirePermission("viewer")), service: SharingService = Depends(get_sharing_service)):
    """共有情報を取得（viewer 以上）"""
    try:
        return service.get_sharing(app_name)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/users")
async def add_user_sharing(app_name: str, body: ShareUserRequest, user=Depends(RequirePermission("owner")), service: SharingService = Depends(get_sharing_service)):
    """ユーザーに権限を付与（owner 以上）"""
    try:
        service.add_user_sharing(app_name, body.user_id, body.permission)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/users/{user_id}")
async def remove_user_sharing(app_name: str, user_id: str, user=Depends(RequirePermission("owner")), service: SharingService = Depends(get_sharing_service)):
    """ユーザーの権限を削除（owner 以上）"""
    try:
        service.remove_user_sharing(app_name, user_id)
        return {"ok": True}
    except LastOwnerError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/groups")
async def add_group_sharing(app_name: str, body: ShareGroupRequest, user=Depends(RequirePermission("owner")), service: SharingService = Depends(get_sharing_service)):
    """グループに権限を付与（owner 以上）"""
    try:
        service.add_group_sharing(app_name, body.group_id, body.permission)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/groups/{group_id}")
async def remove_group_sharing(app_name: str, group_id: str, user=Depends(RequirePermission("owner")), service: SharingService = Depends(get_sharing_service)):
    """グループの権限を削除（owner 以上）"""
    try:
        service.remove_group_sharing(app_name, group_id)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/all")
async def share_with_all(app_name: str, user=Depends(RequirePermission("owner")), service: SharingService = Depends(get_sharing_service)):
    """全ユーザー（all グループ）に共有（owner 以上）"""
    try:
        service.share_with_all(app_name)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(404, str(e))
