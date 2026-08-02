import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel

from errors import register_error_handlers
from exceptions import (
    NotFoundError, BadRequestError, ForbiddenError, ConflictError,
    LastOwnerError, EndpointNotReadyError, ResponseParseError,
)

# main.py は import 時に全 service（AWS クライアント）を構築するため import せず、
# ハンドラだけを適用した最小アプリで検証する。
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
register_error_handlers(app)


class _Body(BaseModel):
    name: str


@app.get("/http-string")
def _http_string():
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/http-dict")
def _http_dict():
    raise HTTPException(status_code=503, detail={"error": "endpoint_not_ready", "message": "起動中です"})


@app.get("/http-500")
def _http_500():
    raise HTTPException(status_code=500, detail="Error: secret internal detail")


@app.get("/http-500-dict")
def _http_500_dict():
    # error コードを持たない 5xx dict はマスクされるべき（内部情報を漏らさない）
    raise HTTPException(status_code=500, detail={"message": "secret dict detail"})


@app.get("/boom")
def _boom():
    raise RuntimeError("secret crash detail")


@app.post("/validate")
def _validate(body: _Body):
    return {"ok": body.name}


# --- domain 例外の各種 ---
@app.get("/err/not-found")
def _err_not_found():
    raise NotFoundError("見つかりません")


@app.get("/err/bad-request")
def _err_bad_request():
    raise BadRequestError("不正な入力です")


@app.get("/err/forbidden")
def _err_forbidden():
    raise ForbiddenError("権限がありません")


@app.get("/err/conflict")
def _err_conflict():
    raise ConflictError("競合しています")


@app.get("/err/last-owner")
def _err_last_owner():
    raise LastOwnerError("最後のオーナーは削除できません")


@app.get("/err/endpoint-not-ready")
def _err_endpoint_not_ready():
    raise EndpointNotReadyError("Endpoint warming up")


@app.get("/err/response-parse")
def _err_response_parse():
    raise ResponseParseError("Failed to parse: secret raw response (stopReason=max_tokens)")


@app.get("/err/value-error")
def _err_value_error():
    # domain 例外でない素の ValueError は 500 になる（ValueError 専用ハンドラは無い）
    raise ValueError("some library value error")


client = TestClient(app, raise_server_exceptions=False)


def test_http_string_detail():
    r = client.get("/http-string")
    assert r.status_code == 404
    assert r.json() == {"detail": "Image not found", "code": None}


def test_endpoint_not_ready_dict_preserves_code():
    r = client.get("/http-dict")
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "endpoint_not_ready"
    assert body["detail"] == "起動中です"


def test_500_hides_internal_detail():
    r = client.get("/http-500")
    assert r.status_code == 500
    body = r.json()
    assert body["code"] == "internal_error"
    assert "secret" not in body["detail"]  # 内部情報がクライアントに漏れない


def test_500_dict_without_error_code_is_masked():
    r = client.get("/http-500-dict")
    assert r.status_code == 500
    body = r.json()
    assert body["code"] == "internal_error"
    assert "secret" not in body["detail"]  # error コード無しの 5xx dict も漏らさない


def test_unhandled_exception_is_masked():
    r = client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["code"] == "internal_error"
    assert "secret" not in body["detail"]


def test_domain_5xx_without_code_hides_internal_detail():
    # 調査用の情報を持つ 5xx は code を持たないので、詳細をクライアントに返さない
    r = client.get("/err/response-parse")
    assert r.status_code == 502
    body = r.json()
    assert body["code"] == "internal_error"
    assert "secret" not in body["detail"]
    assert "stopReason" not in body["detail"]


def test_validation_error_becomes_string():
    r = client.post("/validate", json={})  # name 欠落
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_error"
    assert isinstance(body["detail"], str)  # 配列でなく文字列
    assert body["detail"]  # 非空


def test_framework_404_has_code_key():
    # 未マッチルート（純 starlette の 404）も統一形になる
    r = client.get("/no-such-route")
    assert r.status_code == 404
    assert "code" in r.json()


def test_cors_header_present_on_error():
    r = client.get("/http-string", headers={"Origin": "https://example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"


# --- domain 例外 → (status, code) マッピング ---
@pytest.mark.parametrize("path,status,code", [
    ("/err/not-found", 404, "not_found"),
    ("/err/bad-request", 400, "bad_request"),
    ("/err/forbidden", 403, "forbidden"),
    ("/err/conflict", 409, "conflict"),
    ("/err/last-owner", 400, "last_owner"),
])
def test_domain_exception_mapping(path, status, code):
    r = client.get(path)
    assert r.status_code == status
    body = r.json()
    assert body["code"] == code
    assert isinstance(body["detail"], str) and body["detail"]


def test_endpoint_not_ready_via_app_error_preserves_code():
    # 503 でも code を持つ AppError は message/code を保持（FE 契約）
    r = client.get("/err/endpoint-not-ready")
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "endpoint_not_ready"
    assert body["detail"] == "Endpoint warming up"


def test_bare_value_error_becomes_500():
    # ValueError 専用ハンドラは無いので想定外の 500 になる（中身は隠す）
    r = client.get("/err/value-error")
    assert r.status_code == 500
    body = r.json()
    assert body["code"] == "internal_error"
    assert "library value error" not in body["detail"]
