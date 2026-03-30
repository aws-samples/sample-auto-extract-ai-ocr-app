"""後方互換: from utils.auth import ... の既存 import を維持

実体は dependencies/auth.py に移動済み。
"""
from dependencies.auth import (  # noqa: F401
    get_cognito_sub, get_current_user, get_usecase_permission,
    get_permitted_app_names, require_auth, require_admin,
    RequireRole, RequirePermission,
)
