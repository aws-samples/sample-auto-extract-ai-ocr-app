"""ログインユーザーの DSQL 同期（POST /user/sync の中身）のテスト。

想定している正しい挙動:
- クレームから作ったプロフィールで users を同期し、同期後のユーザー詳細を返す。
- `custom:idp_group` があるときだけ IdP グループを同期する。無いユーザー（IdP 連携していない）
  ではグループ同期を呼ばない。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

import services.user_service as user_service_mod
from services.user_service import UserService


@pytest.fixture
def repos(monkeypatch):
    calls = {"user": [], "groups": []}
    detail = {"id": "u-1", "email": "a@example.com", "display_name": "A", "department": "", "role": "reader"}

    def sync_user(profile):
        calls["user"].append(profile)
        return "u-1"

    monkeypatch.setattr(user_service_mod.user_repository, "sync_user_from_idp", sync_user)
    monkeypatch.setattr(
        user_service_mod.user_repository, "get_user_detail_by_cognito_sub",
        lambda sub: detail if sub == "sub-1" else None,
    )
    monkeypatch.setattr(
        user_service_mod.group_repository, "sync_idp_groups",
        lambda user_id, groups: calls["groups"].append((user_id, groups)),
    )
    calls["detail"] = detail
    return calls


def test_syncs_user_and_returns_detail(repos):
    result = UserService().sync_from_claims({"sub": "sub-1", "email": "a@example.com"})

    assert result == repos["detail"]
    assert [p.cognito_sub for p in repos["user"]] == ["sub-1"]


def test_syncs_idp_groups_for_the_synced_user(repos):
    UserService().sync_from_claims({"sub": "sub-1", "custom:idp_group": "g1,g2"})

    assert repos["groups"] == [("u-1", ["g1", "g2"])]


def test_skips_group_sync_without_idp_group_claim(repos):
    UserService().sync_from_claims({"sub": "sub-1"})

    assert repos["groups"] == []
