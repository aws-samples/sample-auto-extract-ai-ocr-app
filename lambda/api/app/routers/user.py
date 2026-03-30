"""ユーザー API"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from utils.auth import require_auth, get_cognito_sub
from fastapi import Request

from services.user_service import UserService

router = APIRouter(prefix="/user", tags=["User"])

_user_service = UserService()


class UpdateProfileRequest(BaseModel):
    display_name: str


@router.get("/me")
async def get_me(request: Request, user=Depends(require_auth)):
    sub = get_cognito_sub(request)
    detail = _user_service.get_me(sub)
    return {"user": dict(detail) if detail else None}


@router.patch("/me")
async def update_me(body: UpdateProfileRequest, request: Request, user=Depends(require_auth)):
    sub = get_cognito_sub(request)
    if not _user_service.update_display_name(sub, body.display_name.strip()):
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@router.get("/stars")
async def get_stars(request: Request, user=Depends(require_auth)):
    sub = get_cognito_sub(request)
    return {"stars": _user_service.get_stars(sub)}


@router.put("/stars/{app_name}")
async def add_star(app_name: str, request: Request, user=Depends(require_auth)):
    sub = get_cognito_sub(request)
    _user_service.add_star(sub, app_name)
    return {"ok": True}


@router.delete("/stars/{app_name}")
async def remove_star(app_name: str, request: Request, user=Depends(require_auth)):
    sub = get_cognito_sub(request)
    _user_service.remove_star(sub, app_name)
    return {"ok": True}


@router.get("/search")
async def search_users_and_groups(q: str = "", user=Depends(require_auth)):
    """ユーザーとグループを検索（共有設定用）"""
    if not q or len(q) < 1:
        return {"users": [], "groups": []}
    return _user_service.search(q)
