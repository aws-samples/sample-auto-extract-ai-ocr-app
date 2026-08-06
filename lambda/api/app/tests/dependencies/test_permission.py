"""ユースケース権限とシステムロールの判定テスト。

想定している正しい挙動:
- システムロールは reader < author < admin。admin はユースケース権限を問わず全て通る。
- ユースケース権限は viewer < editor < owner。要求レベルに足りなければ 403。
- reader ロールは編集系（editor 以上）を要求されたら、そのユースケースに
  editor/owner 権限を持っていても 403（読むだけのロールとして扱う）。
- 権限レコードが無いユーザーは 403（デフォルト拒否）。
- ルートに適用する `RequirePermission` は、パスの {app_name} を対象に上記を判定する。
  {app_name} を持たないルートに付けるのは設定ミスなので、403 ではなく実行時エラーにする
  （誰でも通ってしまう状態を黙って作らない）。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from fastapi import HTTPException

import dependencies.auth as auth_mod
from dependencies.auth import check_usecase_permission, RequireRole, RequirePermission


@pytest.fixture
def granted(monkeypatch):
    """ユーザーに与えるユースケース権限を差し替える。"""
    state = {"permission": None}
    monkeypatch.setattr(
        auth_mod, "get_usecase_permission",
        lambda user_id, app_name: state["permission"],
    )
    return state


def _user(role):
    return {"id": 1, "role": role}


class TestCheckUsecasePermission:
    def test_admin_passes_without_any_usecase_permission(self, granted):
        granted["permission"] = None
        check_usecase_permission(_user("admin"), "app", "owner")

    def test_no_permission_is_forbidden(self, granted):
        granted["permission"] = None
        with pytest.raises(HTTPException) as e:
            check_usecase_permission(_user("author"), "app", "viewer")
        assert e.value.status_code == 403

    @pytest.mark.parametrize("permission", ["viewer", "editor", "owner"])
    def test_any_permission_satisfies_viewer(self, granted, permission):
        granted["permission"] = permission
        check_usecase_permission(_user("author"), "app", "viewer")

    def test_viewer_permission_cannot_edit(self, granted):
        granted["permission"] = "viewer"
        with pytest.raises(HTTPException) as e:
            check_usecase_permission(_user("author"), "app", "editor")
        assert e.value.status_code == 403

    def test_editor_permission_cannot_reach_owner(self, granted):
        granted["permission"] = "editor"
        with pytest.raises(HTTPException) as e:
            check_usecase_permission(_user("author"), "app", "owner")
        assert e.value.status_code == 403

    def test_owner_permission_satisfies_editor(self, granted):
        granted["permission"] = "owner"
        check_usecase_permission(_user("author"), "app", "editor")

    def test_reader_role_cannot_edit_even_with_owner_permission(self, granted):
        granted["permission"] = "owner"
        with pytest.raises(HTTPException) as e:
            check_usecase_permission(_user("reader"), "app", "editor")
        assert e.value.status_code == 403

    def test_reader_role_can_view(self, granted):
        granted["permission"] = "viewer"
        check_usecase_permission(_user("reader"), "app", "viewer")


class TestRequireRole:
    @pytest.mark.parametrize("role", ["author", "admin"])
    def test_higher_roles_meet_reader_requirement(self, role):
        assert RequireRole("reader")(_user(role)) == _user(role)

    def test_reader_cannot_meet_author_requirement(self):
        with pytest.raises(HTTPException) as e:
            RequireRole("author")(_user("reader"))
        assert e.value.status_code == 403

    def test_author_cannot_meet_admin_requirement(self):
        with pytest.raises(HTTPException) as e:
            RequireRole("admin")(_user("author"))
        assert e.value.status_code == 403

    def test_admin_meets_admin_requirement(self):
        assert RequireRole("admin")(_user("admin")) == _user("admin")

    def test_unknown_role_is_forbidden(self):
        with pytest.raises(HTTPException) as e:
            RequireRole("author")(_user("guest"))
        assert e.value.status_code == 403


class _Request:
    """path_params だけを持つ Request の代役。"""

    def __init__(self, **path_params):
        self.path_params = path_params


class TestRequirePermission:
    def test_checks_permission_for_the_app_in_the_path(self, granted):
        granted["permission"] = "editor"
        user = _user("author")

        assert RequirePermission("editor")(_Request(app_name="app"), user) == user

    def test_insufficient_permission_is_forbidden(self, granted):
        granted["permission"] = "viewer"

        with pytest.raises(HTTPException) as e:
            RequirePermission("editor")(_Request(app_name="app"), _user("author"))
        assert e.value.status_code == 403

    def test_path_without_app_name_is_a_wiring_error(self, granted):
        # 403 ではなく実行時エラー。権限を見ないまま通す状態を作らせない
        granted["permission"] = None

        with pytest.raises(RuntimeError):
            RequirePermission("viewer")(_Request(image_id="img-1"), _user("author"))
