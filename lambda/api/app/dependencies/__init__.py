"""FastAPI Depends 注入パッケージ

services は dependencies.services から直接 import すること。
auth は dependencies.auth から直接 import するか、ここから import 可能。

注意: dependencies.services は services パッケージに依存するため、
ここで re-export すると循環 import が発生する。
"""
from .auth import (
    get_cognito_sub, get_current_user, get_usecase_permission,
    get_permitted_app_names, require_user, check_usecase_permission,
    RequireRole, RequirePermission, RequireImagePermission,
)

__all__ = [
    "get_cognito_sub", "get_current_user", "get_usecase_permission",
    "get_permitted_app_names", "require_user", "check_usecase_permission",
    "RequireRole", "RequirePermission", "RequireImagePermission",
]
