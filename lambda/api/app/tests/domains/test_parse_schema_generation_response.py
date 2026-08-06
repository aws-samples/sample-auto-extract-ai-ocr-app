"""帳票からスキーマを自動生成したときの応答パースのテスト。

想定している正しい挙動:
- 応答は最終的に `{"fields": [...]}` の形に正規化する。
- コードフェンス付きなら中身だけを、無ければ全体を JSON としてパースする。
- モデルが `fields` で包み忘れて配列だけを返した場合は包んで救済する。
- JSON として読めない応答、`fields` キーを持たないオブジェクトは `ResponseParseError` にする。

`fields` の中身が妥当かはここでは見ない（キーの有無だけを判定する）。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from domains.schema_generator import parse_schema_generation_response
from exceptions import ResponseParseError

FIELD = {"name": "total", "display_name": "合計", "type": "string"}


class TestParseSchemaGenerationResponse:
    def test_bare_json_object(self):
        raw = '{"fields": [{"name": "total", "display_name": "合計", "type": "string"}]}'
        assert parse_schema_generation_response(raw) == {"fields": [FIELD]}

    def test_json_in_code_fence(self):
        raw = '```json\n{"fields": [{"name": "total", "display_name": "合計", "type": "string"}]}\n```'
        assert parse_schema_generation_response(raw) == {"fields": [FIELD]}

    def test_bare_array_is_wrapped(self):
        raw = '[{"name": "total", "display_name": "合計", "type": "string"}]'
        assert parse_schema_generation_response(raw) == {"fields": [FIELD]}

    def test_object_without_fields_key_raises(self):
        with pytest.raises(ResponseParseError):
            parse_schema_generation_response('{"foo": 1}')

    def test_invalid_json_raises(self):
        with pytest.raises(ResponseParseError):
            parse_schema_generation_response("スキーマを生成できませんでした")

    @pytest.mark.parametrize("raw", ["42", '"text"', "null"])
    def test_scalar_json_raises(self, raw):
        # JSON として読めてもオブジェクトや配列でなければスキーマにならない
        with pytest.raises(ResponseParseError):
            parse_schema_generation_response(raw)
