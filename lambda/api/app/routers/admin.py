"""管理者 API エンドポイント（DSQL RBAC）"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Literal
import logging

from utils.auth import require_admin
from services.admin_service import AdminService
from services.upload_service import UploadService
from dependencies.services import get_upload_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])

_admin_service = AdminService()


# ========================================
# Users
# ========================================
class UserRoleUpdate(BaseModel):
    role: Literal["admin", "author", "reader"]


@router.get("/users")
async def list_users(user=Depends(require_admin)):
    return {"users": _admin_service.list_users()}


@router.patch("/users/{user_id}/role")
async def update_user_role(user_id: str, body: UserRoleUpdate, user=Depends(require_admin)):
    if not _admin_service.update_user_role(user_id, body.role):
        raise HTTPException(404, "User not found")
    return {"ok": True}


# ========================================
# Groups
# ========================================
class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None


class GroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class GroupMemberUpdate(BaseModel):
    user_ids: list[str]


@router.get("/groups")
async def list_groups(user=Depends(require_admin)):
    return {"groups": _admin_service.list_groups()}


@router.post("/groups")
async def create_group(body: GroupCreate, user=Depends(require_admin)):
    gid = _admin_service.create_group(body.name, body.description)
    return {"id": gid}


@router.get("/groups/{group_id}/members")
async def get_group_members(group_id: str, user=Depends(require_admin)):
    return {"members": _admin_service.get_group_members(group_id)}


@router.delete("/groups/{group_id}")
async def delete_group(group_id: str, user=Depends(require_admin)):
    """グループを削除する（auto グループは削除不可）"""
    try:
        _admin_service.delete_group(group_id)
        return {"ok": True}
    except ValueError as e:
        detail = str(e)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status, detail)


@router.patch("/groups/{group_id}")
async def update_group(group_id: str, body: GroupUpdate, user=Depends(require_admin)):
    """グループの名前・説明を更新する（auto グループは編集不可）"""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")
    try:
        if not _admin_service.update_group(group_id, **updates):
            raise HTTPException(404, "Group not found")
        return {"ok": True}
    except ValueError as e:
        detail = str(e)
        status = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status, detail)


@router.put("/groups/{group_id}/members")
async def update_group_members(group_id: str, body: GroupMemberUpdate, user=Depends(require_admin)):
    _admin_service.update_group_members(group_id, body.user_ids)
    return {"ok": True}


# ========================================
# Usecases
# ========================================
@router.get("/usecases")
async def list_usecases(user=Depends(require_admin)):
    return {"usecases": _admin_service.list_usecases()}


@router.get("/usecases/{usecase_id}/permissions")
async def get_usecase_permissions(usecase_id: str, user=Depends(require_admin)):
    return _admin_service.get_usecase_permissions(usecase_id)


# ========================================
# Tools
# ========================================
class ToolCreate(BaseModel):
    name: str
    tool_name: str
    description: Optional[str] = None


class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


@router.get("/tools")
async def list_tools(user=Depends(require_admin)):
    return {"tools": _admin_service.list_tools()}


@router.get("/tools/{tool_id}/permissions")
async def get_tool_permissions(tool_id: str, user=Depends(require_admin)):
    return _admin_service.get_tool_permissions(tool_id)


@router.post("/tools")
async def create_tool(body: ToolCreate, user=Depends(require_admin)):
    tid = _admin_service.create_tool(body.name, body.tool_name, body.description)
    return {"id": tid}


@router.patch("/tools/{tool_id}")
async def update_tool(tool_id: str, body: ToolUpdate, user=Depends(require_admin)):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")
    if not _admin_service.update_tool(tool_id, updates):
        raise HTTPException(404, "Tool not found")
    return {"ok": True}


class ToolUserBody(BaseModel):
    user_id: str


class ToolGroupBody(BaseModel):
    group_id: str


@router.post("/tools/{tool_id}/users")
async def add_tool_user(tool_id: str, body: ToolUserBody, user=Depends(require_admin)):
    """ツールにユーザー権限を追加"""
    _admin_service.add_tool_user(tool_id, body.user_id)
    return {"ok": True}


@router.delete("/tools/{tool_id}/users/{user_id}")
async def remove_tool_user(tool_id: str, user_id: str, user=Depends(require_admin)):
    """ツールからユーザー権限を削除"""
    _admin_service.remove_tool_user(tool_id, user_id)
    return {"ok": True}


@router.post("/tools/{tool_id}/groups")
async def add_tool_group(tool_id: str, body: ToolGroupBody, user=Depends(require_admin)):
    """ツールにグループ権限を追加"""
    _admin_service.add_tool_group(tool_id, body.group_id)
    return {"ok": True}


@router.delete("/tools/{tool_id}/groups/{group_id}")
async def remove_tool_group(tool_id: str, group_id: str, user=Depends(require_admin)):
    """ツールからグループ権限を削除"""
    _admin_service.remove_tool_group(tool_id, group_id)
    return {"ok": True}


# ========================================
# Images (admin: 全履歴)
# ========================================
@router.get("/images")
async def list_all_images(app_name: str = None, user=Depends(require_admin), service: UploadService = Depends(get_upload_service)):
    """全ユーザーの画像一覧（admin 用）"""
    return await service.get_images_list(app_name, uploaded_by=None)
