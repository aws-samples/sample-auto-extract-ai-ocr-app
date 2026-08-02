"""Bedrock レスポンスからテキスト / JSON を取り出す純関数のテスト。

想定している正しい挙動: どちらも「取れなければ空を返す」。取れなかったことを
失敗として扱うかどうかは呼び出し側（サービス層）が決める。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.bedrock import parse_converse_response, extract_json_from_response


class TestParseConverseResponse:
    def test_returns_first_text_block(self):
        response = {"output": {"message": {"content": [{"text": "hello"}]}}}
        assert parse_converse_response(response) == "hello"

    def test_empty_content_returns_empty_string(self):
        response = {"output": {"message": {"content": []}}}
        assert parse_converse_response(response) == ""

    def test_missing_key_returns_empty_string(self):
        assert parse_converse_response({"output": {}}) == ""


class TestExtractJsonFromResponse:
    def test_extracts_json_object(self):
        assert extract_json_from_response('前置き {"a": 1} 後置き') == {"a": 1}

    def test_no_json_returns_empty_dict(self):
        assert extract_json_from_response("JSONはありません") == {}

    def test_broken_json_returns_empty_dict(self):
        # 括弧は閉じているが JSON として不正（末尾カンマ）
        assert extract_json_from_response('{"a": 1,}') == {}
