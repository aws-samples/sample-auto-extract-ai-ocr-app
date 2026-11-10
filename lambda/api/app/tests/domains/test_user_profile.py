"""ID トークンのクレームからユーザー同期用プロフィールを作る処理のテスト。

想定している正しい挙動:
- `custom:idp_group` はカンマ区切り。前後の空白・空要素・重複を落とし、出現順を保つ。
- `name` が無い / 空なら表示名はメールアドレスにする。
- 欠けているクレームは空文字として扱う（None を DSQL に書かない）。
- 既存ユーザーの更新要否は email / department だけで判定する。display_name や role が
  IdP の値と違っても更新しない（利用者・管理者が変えた値をログインのたびに戻さない）。
- department は NULL と空文字を同じものとして扱う（未設定のユーザーを毎回更新しない）。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from domains.user_profile import IdpUserProfile, needs_profile_update, profile_from_claims


def _profile(email="a@example.com", department=""):
    return IdpUserProfile(
        cognito_sub="sub-1", email=email, display_name="A", department=department, idp_groups=(),
    )


class TestProfileFromClaims:
    def test_maps_all_claims(self):
        profile = profile_from_claims({
            "sub": "sub-1",
            "email": "a@example.com",
            "name": "Alice",
            "custom:department": "Sales",
            "custom:idp_group": "g1,g2",
        })

        assert profile == IdpUserProfile(
            cognito_sub="sub-1",
            email="a@example.com",
            display_name="Alice",
            department="Sales",
            idp_groups=("g1", "g2"),
        )

    def test_idp_groups_are_trimmed_deduplicated_and_ordered(self):
        profile = profile_from_claims({"sub": "s", "custom:idp_group": " g2 , g1,,g2, "})

        assert profile.idp_groups == ("g2", "g1")

    def test_missing_idp_group_means_no_groups(self):
        assert profile_from_claims({"sub": "s"}).idp_groups == ()

    def test_display_name_falls_back_to_email(self):
        assert profile_from_claims({"sub": "s", "email": "a@example.com"}).display_name == "a@example.com"
        assert profile_from_claims({"sub": "s", "email": "a@example.com", "name": ""}).display_name == "a@example.com"

    def test_missing_claims_become_empty_strings(self):
        profile = profile_from_claims({"sub": "s"})

        assert profile.email == ""
        assert profile.display_name == ""
        assert profile.department == ""


class TestNeedsProfileUpdate:
    def test_unchanged_row_is_not_updated(self):
        existing = {"email": "a@example.com", "department": "Sales"}

        assert needs_profile_update(existing, _profile(department="Sales")) is False

    def test_email_change_is_updated(self):
        existing = {"email": "old@example.com", "department": ""}

        assert needs_profile_update(existing, _profile()) is True

    def test_department_change_is_updated(self):
        existing = {"email": "a@example.com", "department": "Sales"}

        assert needs_profile_update(existing, _profile(department="Dev")) is True

    def test_null_department_equals_empty_claim(self):
        existing = {"email": "a@example.com", "department": None}

        assert needs_profile_update(existing, _profile(department="")) is False

    def test_display_name_and_role_differences_are_ignored(self):
        existing = {
            "email": "a@example.com", "department": "",
            "display_name": "利用者が付けた名前", "role": "admin",
        }

        assert needs_profile_update(existing, _profile()) is False
