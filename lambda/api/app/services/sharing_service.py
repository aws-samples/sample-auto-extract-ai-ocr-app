"""共有設定サービス"""
import logging

from exceptions import LastOwnerError, NotFoundError
from repositories import usecase_repository, group_repository

logger = logging.getLogger(__name__)


class SharingService:
    def __init__(self):
        pass

    def _get_usecase_id(self, app_name: str) -> str:
        uc = usecase_repository.get_usecase_by_app_name(app_name)
        if not uc:
            raise NotFoundError("ユースケースが見つかりません")
        return str(uc["id"])

    def get_sharing(self, app_name: str) -> dict:
        uc_id = self._get_usecase_id(app_name)
        owners = usecase_repository.get_usecase_owners(uc_id)
        users = usecase_repository.get_usecase_user_permissions(uc_id)
        groups = usecase_repository.get_usecase_group_permissions(uc_id)
        return {"owners": owners, "users": users, "groups": groups}

    def add_user_sharing(self, app_name: str, user_id: str, permission: str) -> None:
        uc_id = self._get_usecase_id(app_name)
        usecase_repository.upsert_user_permission(user_id, uc_id, permission)

    def remove_user_sharing(self, app_name: str, user_id: str) -> None:
        uc_id = self._get_usecase_id(app_name)
        deleted = usecase_repository.delete_user_permission_safe(user_id, uc_id)
        if not deleted:
            raise LastOwnerError("最後のオーナーは削除できません")

    def add_group_sharing(self, app_name: str, group_id: str, permission: str) -> None:
        uc_id = self._get_usecase_id(app_name)
        usecase_repository.upsert_group_permission(group_id, uc_id, permission)

    def remove_group_sharing(self, app_name: str, group_id: str) -> None:
        uc_id = self._get_usecase_id(app_name)
        usecase_repository.delete_group_permission(group_id, uc_id)

    def share_with_all(self, app_name: str) -> None:
        uc_id = self._get_usecase_id(app_name)
        all_group = group_repository.get_group_by_name("all")
        if not all_group:
            raise NotFoundError("all グループが見つかりません")
        usecase_repository.upsert_group_permission(str(all_group["id"]), uc_id, "viewer")
