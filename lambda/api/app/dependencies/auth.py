"""認証認可ユーティリティ（FastAPI Depends ベース）

- RequireRole: システム全体のロール（admin/author/reader）をチェック
- RequirePermission: 特定ユースケースに対する権限（owner/editor/viewer）をチェック
"""
from fastapi import Request, HTTPException, Depends
import json
from repositories.usecase_repository import get_user_max_permission, get_permitted_app_names as _get_permitted_app_names
from repositories.user_repository import get_user_by_cognito_sub

_LEVEL_RANK = {"viewer": 1, "editor": 2, "owner": 3}   # ユースケース権限
_ROLE_RANK = {"reader": 1, "author": 2, "admin": 3}     # システムロール


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


class RequireRole:
    """指定ロール以上を要求する Depends クラス"""
    def __init__(self, min_role: str):
        self.min_role = min_role

    def __call__(self, user: dict = Depends(require_auth)) -> dict:
        # _ROLE_RANK で比較: reader(1) < author(2) < admin(3)
        if _ROLE_RANK.get(user["role"], 0) < _ROLE_RANK.get(self.min_role, 0):
            raise HTTPException(status_code=403, detail=f"Forbidden: requires {self.min_role} role")
        return user


class RequirePermission:
    """ユースケース単位の権限チェック Depends クラス。
    パスパラメータ app_name を自動取得し、user_usecases / group_usecases から
    最大権限を解決する。admin ロールは全権限スキップ。"""
    def __init__(self, min_level: str):
        self.min_level = min_level

    def __call__(self, app_name: str, user: dict = Depends(require_auth)) -> dict:
        # admin は全ユースケースにフルアクセス
        if user["role"] == "admin":
            return user

        required = _LEVEL_RANK.get(self.min_level, 0)

        # reader ロールは viewer 権限までしか行使できない
        if user["role"] == "reader" and required > _LEVEL_RANK["viewer"]:
            raise HTTPException(status_code=403, detail="Forbidden: reader role cannot edit")

        # user_usecases + group_usecases から最大権限を取得し、_LEVEL_RANK で比較:
        # viewer(1) < editor(2) < owner(3)
        perm = get_usecase_permission(str(user["id"]), app_name)
        if not perm or _LEVEL_RANK.get(perm, 0) < required:
            raise HTTPException(status_code=403, detail=f"Forbidden: requires {self.min_level} permission")
        return user
