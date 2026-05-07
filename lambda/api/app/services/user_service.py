"""ユーザー情報サービス"""
import logging

from repositories import user_repository, group_repository
from repositories import user_preferences_repository

logger = logging.getLogger(__name__)


class UserService:
    def __init__(self):
        pass

    def get_me(self, cognito_sub: str) -> dict | None:
        return user_repository.get_user_detail_by_cognito_sub(cognito_sub)

    def get_stars(self, cognito_sub: str) -> list[str]:
        return user_preferences_repository.get_stars(cognito_sub)

    def add_star(self, cognito_sub: str, app_name: str) -> None:
        user_preferences_repository.add_star(cognito_sub, app_name)

    def remove_star(self, cognito_sub: str, app_name: str) -> None:
        user_preferences_repository.remove_star(cognito_sub, app_name)

    def update_display_name(self, cognito_sub: str, display_name: str) -> bool:
        return user_repository.update_display_name(cognito_sub, display_name)

    def search(self, q: str) -> dict:
        pattern = f"%{q}%"
        users = user_repository.search_users(pattern)
        groups = group_repository.search_groups(pattern)
        return {"users": users, "groups": groups}
