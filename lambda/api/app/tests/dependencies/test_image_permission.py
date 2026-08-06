"""画像単位の権限ガードのテスト（他人の画像を開けないことを守る要）。

実際のリクエスト経路で確認したいので、ガードだけを付けた最小アプリを立てて叩く。

想定している正しい挙動:
- 認証情報が無いリクエストは 401。
- 存在しない image_id は 404。
- 画像がどのユースケースにも紐付いていない場合は 403（判断できないなら拒否）。
- 画像のユースケースに対する権限が足りなければ 403（他人の画像は開けない）。
- 権限が足りていれば通し、取得済みの画像レコードを request.state.image に載せて
  後続で再取得させない。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json

import pytest
from fastapi import FastAPI, Depends, Request
from fastapi.testclient import TestClient

import dependencies.auth as auth_mod
from dependencies.auth import RequireImagePermission
from errors import register_error_handlers

# main.py は import 時に全 service（AWS クライアント）を構築するため import せず、
# ガードだけを付けた最小アプリで検証する。
app = FastAPI()
register_error_handlers(app)


@app.get("/images/{image_id}")
def _read_image(request: Request, user: dict = Depends(RequireImagePermission("viewer"))):
    return {"cached": request.state.image, "user_id": user["id"]}


@app.put("/images/{image_id}")
def _edit_image(user: dict = Depends(RequireImagePermission("editor"))):
    return {"ok": True}


client = TestClient(app)

AUTH_HEADERS = {
    "x-amzn-request-context": json.dumps({"authorizer": {"claims": {"sub": "sub-1"}}})
}


@pytest.fixture
def env(monkeypatch):
    """ユーザー・画像・権限を差し替える。role は非 admin にして権限判定を実際に通す。"""
    state = {
        "user": {"id": 1, "role": "author"},
        "image": {"id": "img-1", "app_name": "app"},
        "permission": "viewer",
    }
    monkeypatch.setattr(auth_mod, "get_user_by_cognito_sub", lambda sub: state["user"])
    monkeypatch.setattr(auth_mod, "get_image", lambda image_id: state["image"])
    monkeypatch.setattr(
        auth_mod, "get_usecase_permission",
        lambda user_id, app_name: state["permission"],
    )
    return state


class TestRequireImagePermission:
    def test_missing_auth_header_is_unauthorized(self, env):
        assert client.get("/images/img-1").status_code == 401

    def test_unknown_user_is_unauthorized(self, env):
        env["user"] = None
        assert client.get("/images/img-1", headers=AUTH_HEADERS).status_code == 401

    def test_missing_image_is_not_found(self, env):
        env["image"] = None
        assert client.get("/images/img-1", headers=AUTH_HEADERS).status_code == 404

    def test_image_without_usecase_is_forbidden(self, env):
        env["image"] = {"id": "img-1"}
        assert client.get("/images/img-1", headers=AUTH_HEADERS).status_code == 403

    def test_no_permission_is_forbidden(self, env):
        env["permission"] = None
        assert client.get("/images/img-1", headers=AUTH_HEADERS).status_code == 403

    def test_viewer_permission_can_read(self, env):
        response = client.get("/images/img-1", headers=AUTH_HEADERS)
        assert response.status_code == 200
        # 取得済みの画像レコードが後続に渡る
        assert response.json()["cached"] == {"id": "img-1", "app_name": "app"}

    def test_viewer_permission_cannot_edit(self, env):
        env["permission"] = "viewer"
        assert client.put("/images/img-1", headers=AUTH_HEADERS).status_code == 403

    def test_editor_permission_can_edit(self, env):
        env["permission"] = "editor"
        assert client.put("/images/img-1", headers=AUTH_HEADERS).status_code == 200

    def test_admin_can_read_without_usecase_permission(self, env):
        env["user"] = {"id": 9, "role": "admin"}
        env["permission"] = None
        assert client.get("/images/img-1", headers=AUTH_HEADERS).status_code == 200
