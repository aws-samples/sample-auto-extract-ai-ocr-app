"""ユーザー API"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from dependencies.auth import require_auth, get_cognito_sub
from fastapi import Request

from services.user_service import UserService
from dependencies.services import get_user_service

router = APIRouter(prefix="/user", tags=["User"])


class UpdateProfileRequest(BaseModel):
    display_name: str


@router.get("/me")
async def get_me(request: Request, user=Depends(require_auth), service: UserService = Depends(get_user_service)):
    sub = get_cognito_sub(request)
    detail = service.get_me(sub)
    return {"user": dict(detail) if detail else None}


@router.patch("/me")
async def update_me(body: UpdateProfileRequest, request: Request, user=Depends(require_auth), service: UserService = Depends(get_user_service)):
    sub = get_cognito_sub(request)
    if not service.update_display_name(sub, body.display_name.strip()):
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@router.get("/stars")
async def get_stars(request: Request, user=Depends(require_auth), service: UserService = Depends(get_user_service)):
    sub = get_cognito_sub(request)
    return {"stars": service.get_stars(sub)}


@router.put("/stars/{app_name}")
async def add_star(app_name: str, request: Request, user=Depends(require_auth), service: UserService = Depends(get_user_service)):
    sub = get_cognito_sub(request)
    service.add_star(sub, app_name)
    return {"ok": True}


@router.delete("/stars/{app_name}")
async def remove_star(app_name: str, request: Request, user=Depends(require_auth), service: UserService = Depends(get_user_service)):
    sub = get_cognito_sub(request)
    service.remove_star(sub, app_name)
    return {"ok": True}


@router.get("/search")
async def search_users_and_groups(q: str = "", user=Depends(require_auth), service: UserService = Depends(get_user_service)):
    """ユーザーとグループを検索（共有設定用）"""
    if not q or len(q) < 1:
        return {"users": [], "groups": []}
    return service.search(q)
