"""管理者操作サービス"""
import logging

from repositories import user_repository, group_repository, usecase_repository, tool_repository

logger = logging.getLogger(__name__)


class NotFoundError(Exception):
    """リソースが見つからない場合のエラー"""
    pass


class AdminService:
    def __init__(self):
        pass

    # ---- Users ----
    def list_users(self) -> list[dict]:
        users = user_repository.list_users()
        # 各ユーザーの所属グループ名を付与
        user_ids = [str(u["id"]) for u in users]
        group_map = group_repository.get_user_group_names(user_ids)
        for u in users:
            u["groups"] = group_map.get(str(u["id"]), [])
        return users

    def update_user_role(self, user_id: str, role: str) -> bool:
        return user_repository.update_user_role(user_id, role)

    # ---- Groups ----
    def list_groups(self) -> list[dict]:
        return group_repository.list_groups()

    def create_group(self, name: str, description: str | None = None) -> str:
        return group_repository.create_group(name, description)

    def get_group(self, group_id: str) -> dict | None:
        return group_repository.get_group(group_id)

    def get_group_members(self, group_id: str) -> list[dict]:
        return group_repository.get_group_members(group_id)

    def delete_group(self, group_id: str) -> None:
        """グループを削除する（auto グループは削除不可）"""
        group = group_repository.get_group(group_id)
        if not group:
            raise NotFoundError("Group not found")
        if group["source"] == "auto":
            raise ValueError("Cannot delete auto-managed group")
        group_repository.delete_group(group_id)

    def update_group(self, group_id: str, name: str = None, description: str = None) -> bool:
        """グループの名前・説明を更新する（auto グループは編集不可）"""
        group = group_repository.get_group(group_id)
        if not group:
            raise NotFoundError("Group not found")
        if group["source"] == "auto":
            raise ValueError("Cannot edit auto-managed group")
        return group_repository.update_group(group_id, name=name, description=description)

    def update_group_members(self, group_id: str, user_ids: list[str]) -> None:
        group_repository.update_group_members(group_id, user_ids)

    # ---- Usecases ----
    def list_usecases(self) -> list[dict]:
        return usecase_repository.list_usecases()

    def get_usecase_permissions(self, usecase_id: str) -> dict:
        return {
            "users": usecase_repository.get_usecase_user_permissions(usecase_id),
            "groups": usecase_repository.get_usecase_group_permissions(usecase_id),
        }

    # ---- Tools ----
    def list_tools(self) -> list[dict]:
        return tool_repository.list_tools()

    def create_tool(self, name: str, tool_name: str, description: str | None = None) -> str:
        return tool_repository.create_tool(name, tool_name, description)

    def update_tool(self, tool_id: str, updates: dict) -> bool:
        return tool_repository.update_tool(tool_id, updates)

    def get_tool_permissions(self, tool_id: str) -> dict:
        return {
            "users": tool_repository.get_tool_user_permissions(tool_id),
            "groups": tool_repository.get_tool_group_permissions(tool_id),
            "usecases": tool_repository.get_tool_usecase_permissions(tool_id),
        }

    # ---- Tool permissions ----
    def add_tool_user(self, tool_id: str, user_id: str) -> None:
        tool_repository.add_tool_user(tool_id, user_id)

    def remove_tool_user(self, tool_id: str, user_id: str) -> None:
        tool_repository.remove_tool_user(tool_id, user_id)

    def add_tool_group(self, tool_id: str, group_id: str) -> None:
        tool_repository.add_tool_group(tool_id, group_id)

    def remove_tool_group(self, tool_id: str, group_id: str) -> None:
        tool_repository.remove_tool_group(tool_id, group_id)
