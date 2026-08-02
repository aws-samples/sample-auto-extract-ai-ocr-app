"""LLM 応答から抽出結果を取り出すパース処理のテスト。

想定している正しい挙動:
- `parse_extraction_response` は応答 JSON から `extracted_data`（抽出値）と
  `indices`（OCR 語との対応）を取り出す。両方揃っているときだけ成功。
- 応答が期待形式でない場合は `ExtractionParseError` を投げる。
  「エラー内容を戻り値に入れて返す」ことはしない（呼び出し元が失敗を成功と誤認するため）。
- 抽出値の中身は LLM が返したユースケース定義そのままなので、内容の検証はしない。
  `error` という名前のフィールドが含まれていても正常な抽出結果として扱う。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json

import pytest

from domains.extraction_engine import parse_extraction_response, ExtractionParseError


def _payload(extracted_data, indices):
    return json.dumps({"extracted_data": extracted_data, "indices": indices})


class TestParseExtractionResponse:
    def test_bare_json(self):
        raw = _payload({"total": "1000"}, {"total": [3]})
        assert parse_extraction_response(raw) == ({"total": "1000"}, {"total": [3]})

    def test_json_in_code_fence(self):
        raw = "```json\n" + _payload({"total": "1000"}, {"total": [3]}) + "\n```"
        assert parse_extraction_response(raw) == ({"total": "1000"}, {"total": [3]})

    def test_surrounding_prose_is_ignored(self):
        raw = "はい、以下が結果です。\n" + _payload({"a": "1"}, {"a": []}) + "\n以上です。"
        assert parse_extraction_response(raw) == ({"a": "1"}, {"a": []})

    def test_field_named_error_is_still_a_success(self):
        # 抽出対象のフィールド名として "error" は正当に定義できるため、
        # error キーの有無で失敗判定してはいけない
        extracted, mapping = parse_extraction_response(
            _payload({"error": "設備異常"}, {"error": [7]})
        )
        assert extracted == {"error": "設備異常"}
        assert mapping == {"error": [7]}

    def test_missing_indices_raises(self):
        with pytest.raises(ExtractionParseError):
            parse_extraction_response(json.dumps({"extracted_data": {"a": "1"}}))

    def test_missing_extracted_data_raises(self):
        with pytest.raises(ExtractionParseError):
            parse_extraction_response(json.dumps({"indices": {"a": []}}))

    def test_no_json_raises(self):
        with pytest.raises(ExtractionParseError):
            parse_extraction_response("抽出できませんでした")

    def test_broken_json_raises(self):
        # 括弧は閉じているが JSON として不正（末尾カンマ）
        with pytest.raises(ExtractionParseError):
            parse_extraction_response('{"extracted_data": {"a": 1,}, "indices": {}}')
