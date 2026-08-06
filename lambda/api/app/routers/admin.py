"""管理者 API エンドポイント（DSQL RBAC）"""
from fastapi import APIRouter, HTTPException, Depends
import logging

from dependencies.auth import RequireRole
from schemas import UsecaseToolsUpdate
from schemas.admin import (
    UserRoleUpdate, GroupCreate, GroupUpdate, GroupMemberUpdate,
    ToolCreate, ToolUpdate, ToolUserBody, ToolGroupBody,
)
from services.admin_service import AdminService
from services.image_list_service import ImageListService
from dependencies.services import get_image_list_service, get_admin_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])


# ========================================
# Users
# ========================================
@router.get("/users")
async def list_users(user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    return {"users": service.list_users()}


@router.patch("/users/{user_id}/role")
async def update_user_role(user_id: str, body: UserRoleUpdate, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    if not service.update_user_role(user_id, body.role):
        raise HTTPException(404, "User not found")
    return {"ok": True}


# ========================================
# Groups
# ========================================
@router.get("/groups")
async def list_groups(user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    return {"groups": service.list_groups()}


@router.post("/groups")
async def create_group(body: GroupCreate, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    gid = service.create_group(body.name, body.description)
    return {"id": gid}


@router.get("/groups/{group_id}/members")
async def get_group_members(group_id: str, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    return {"members": service.get_group_members(group_id)}


@router.delete("/groups/{group_id}")
async def delete_group(group_id: str, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    """グループを削除する（auto グループは削除不可）"""
    service.delete_group(group_id)
    return {"ok": True}


@router.patch("/groups/{group_id}")
async def update_group(group_id: str, body: GroupUpdate, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    """グループの名前・説明を更新する（auto グループは編集不可）"""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")
    if not service.update_group(group_id, **updates):
        raise HTTPException(404, "Group not found")
    return {"ok": True}


@router.put("/groups/{group_id}/members")
async def update_group_members(group_id: str, body: GroupMemberUpdate, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    service.update_group_members(group_id, body.user_ids)
    return {"ok": True}


# ========================================
# Usecases
# ========================================
@router.get("/usecases")
async def list_usecases(user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    return {"usecases": service.list_usecases()}


@router.get("/usecases/{app_name}/permissions")
async def get_usecase_permissions(app_name: str, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    return service.get_usecase_permissions(app_name)


@router.get("/usecases/{app_name}/tools")
async def get_usecase_tools(app_name: str, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    """ユースケースに紐付くツール一覧"""
    return {"tools": service.get_usecase_tools(app_name)}


@router.put("/usecases/{app_name}/tools")
async def set_usecase_tools(app_name: str, body: UsecaseToolsUpdate, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    """ユースケースのツールを一括設定"""
    service.set_usecase_tools(app_name, body.tool_ids)
    return {"ok": True}


# ========================================
# Tools
# ========================================
@router.get("/tools")
async def list_tools(user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    return {"tools": service.list_tools()}


@router.get("/tools/{tool_id}/permissions")
async def get_tool_permissions(tool_id: str, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    return service.get_tool_permissions(tool_id)


@router.post("/tools")
async def create_tool(body: ToolCreate, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    tid = service.create_tool(body.name, body.description)
    return {"id": tid}


@router.patch("/tools/{tool_id}")
async def update_tool(tool_id: str, body: ToolUpdate, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")
    if not service.update_tool(tool_id, updates):
        raise HTTPException(404, "Tool not found")
    return {"ok": True}


@router.post("/tools/{tool_id}/users")
async def add_tool_user(tool_id: str, body: ToolUserBody, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    """ツールにユーザー権限を追加"""
    service.add_tool_user(tool_id, body.user_id)
    return {"ok": True}


@router.delete("/tools/{tool_id}/users/{user_id}")
async def remove_tool_user(tool_id: str, user_id: str, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    """ツールからユーザー権限を削除"""
    service.remove_tool_user(tool_id, user_id)
    return {"ok": True}


@router.post("/tools/{tool_id}/groups")
async def add_tool_group(tool_id: str, body: ToolGroupBody, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    """ツールにグループ権限を追加"""
    service.add_tool_group(tool_id, body.group_id)
    return {"ok": True}


@router.delete("/tools/{tool_id}/groups/{group_id}")
async def remove_tool_group(tool_id: str, group_id: str, user=Depends(RequireRole("admin")), service: AdminService = Depends(get_admin_service)):
    """ツールからグループ権限を削除"""
    service.remove_tool_group(tool_id, group_id)
    return {"ok": True}


# ========================================
# Images (admin: 全履歴)
# ========================================
@router.get("/images")
async def list_all_images(app_name: str = None, user=Depends(RequireRole("admin")), service: ImageListService = Depends(get_image_list_service)):
    """全ユーザーの画像一覧（admin 用）"""
    return await service.get_images_list(app_name, uploaded_by=None)
