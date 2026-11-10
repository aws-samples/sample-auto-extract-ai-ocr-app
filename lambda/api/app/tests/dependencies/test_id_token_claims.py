"""ID トークンのクレーム取得と、users 未登録でも通す認証のテスト。

想定している正しい挙動:
- クレームは LWA が付ける `x-amzn-request-context` の authorizer.claims から読む。
- ヘッダーが無い / JSON が壊れている / 形が違うときは空として扱う（例外にしない）。
- `require_id_token_claims` は sub があれば users 登録を問わず通し、無ければ 401。
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from fastapi import HTTPException

from dependencies.auth import get_cognito_sub, get_id_token_claims, require_id_token_claims


class _Request:
    """headers だけを持つ Request の代役。"""

    def __init__(self, context=None):
        self.headers = {} if context is None else {"x-amzn-request-context": context}


def _ctx(claims):
    return json.dumps({"authorizer": {"claims": claims}})


class TestGetIdTokenClaims:
    def test_reads_claims_from_request_context(self):
        claims = {"sub": "sub-1", "email": "a@example.com"}

        assert get_id_token_claims(_Request(_ctx(claims))) == claims
        assert get_cognito_sub(_Request(_ctx(claims))) == "sub-1"

    @pytest.mark.parametrize("context", [None, "", "not-json", "[]", json.dumps({"authorizer": {"claims": "x"}})])
    def test_missing_or_malformed_context_is_empty(self, context):
        assert get_id_token_claims(_Request(context)) == {}
        assert get_cognito_sub(_Request(context)) == ""


class TestRequireIdTokenClaims:
    def test_passes_with_sub(self):
        claims = {"sub": "sub-1"}

        assert require_id_token_claims(_Request(_ctx(claims))) == claims

    def test_without_sub_is_unauthorized(self):
        with pytest.raises(HTTPException) as e:
            require_id_token_claims(_Request(_ctx({"email": "a@example.com"})))
        assert e.value.status_code == 401
