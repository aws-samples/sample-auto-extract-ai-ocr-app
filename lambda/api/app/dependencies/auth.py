"""認証認可ユーティリティ（FastAPI Depends ベース）"""
from fastapi import Request, HTTPException, Depends
import json
from repositories.usecase_repository import get_user_max_permission, get_permitted_app_names as _get_permitted_app_names
from repositories.user_repository import get_user_by_cognito_sub

_LEVEL_RANK = {"viewer": 1, "editor": 2, "owner": 3}
_ROLE_RANK = {"reader": 1, "author": 2, "admin": 3}


# ============================================================
# 低レベルヘルパー（Depends 以外からも利用可能）
# ============================================================

def get_cognito_sub(request: Request) -> str:
    """LWA の x-amzn-request-context ヘッダーから cognito_sub を取得"""
    ctx = request.headers.get("x-amzn-request-context", "")
    if ctx:
        try:
            data = json.loads(ctx)
            return data.get("authorizer", {}).get("claims", {}).get("sub", "")
        except (json.JSONDecodeError, AttributeError):
            pass
    return ""


def get_current_user(request: Request) -> dict | None:
    """リクエストから現在のユーザー情報を取得（id, role）"""
    sub = get_cognito_sub(request)
    if not sub:
        return None
    return get_user_by_cognito_sub(sub)


def get_usecase_permission(user_id: str, app_name: str) -> str | None:
    """ユーザーのユースケースに対する最大権限を返す"""
    return get_user_max_permission(user_id, app_name)


def get_permitted_app_names(user_id: str) -> list[str]:
    """ユーザーが何らかの権限を持つ app_name 一覧を返す"""
    return _get_permitted_app_names(user_id)


# ============================================================
# Depends ベースの認証認可
# ============================================================

def require_auth(request: Request) -> dict:
    """認証チェック（ログイン必須）。ユーザー情報を返す。"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def require_admin(user: dict = Depends(require_auth)) -> dict:
    """admin ロール必須"""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user


class RequireRole:
    """指定ロール以上を要求する Depends クラス"""
    def __init__(self, min_role: str):
        self.min_role = min_role

    def __call__(self, user: dict = Depends(require_auth)) -> dict:
        if _ROLE_RANK.get(user["role"], 0) < _ROLE_RANK.get(self.min_role, 0):
            raise HTTPException(status_code=403, detail=f"Forbidden: requires {self.min_role} role")
        return user


class RequirePermission:
    """ユースケース権限チェック Depends クラス。
    パスパラメータ app_name を自動取得する。"""
    def __init__(self, min_level: str):
        self.min_level = min_level

    def __call__(self, app_name: str, user: dict = Depends(require_auth)) -> dict:
        if user["role"] == "admin":
            return user

        required = _LEVEL_RANK.get(self.min_level, 0)

        if user["role"] == "reader" and required > _LEVEL_RANK["viewer"]:
            raise HTTPException(status_code=403, detail="Forbidden: reader role cannot edit")

        perm = get_usecase_permission(str(user["id"]), app_name)
        if not perm or _LEVEL_RANK.get(perm, 0) < required:
            raise HTTPException(status_code=403, detail=f"Forbidden: requires {self.min_level} permission")
        return user
