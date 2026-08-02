"""保存済み OCR 結果を API レスポンスに変換する処理のテスト。

想定している正しい挙動:
- 認識結果が揃っているレコードは、単語以外の付随情報（ページ数など）も含めてそのまま返す。
- 処理が失敗した画像には単語リストを持たないレコードが保存されている。この場合は
  単語 0 件 + エラー内容として返す（単語リストが無いことを理由に 500 にしない）。
- OCR 無効時は OCR 結果自体が無い。この場合も単語 0 件として返す。

単語リストが「ある」場合の中身は検証しない。ここが守るのは単語リストを取れないケースだけで、
単語の各要素が壊れている場合は従来どおりレスポンス組み立てで失敗する。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from services.ocr_service import OcrService

to_ocr_result = OcrService._to_ocr_result


class TestToOcrResult:
    def test_keeps_multipage_fields(self):
        stored = {
            "words": [{"id": 0, "content": "abc"}],
            "pages": [{"page": 1, "words": []}],
            "total_pages": 2,
            "word_count": 1,
        }
        result = to_ocr_result(stored)
        assert len(result.words) == 1
        assert result.total_pages == 2
        assert result.pages == [{"page": 1, "words": []}]
        assert result.error is None

    def test_failed_record_without_words_returns_error(self):
        stored = {"error": "conversion failed", "timestamp": "2026-08-01T00:00:00Z"}
        result = to_ocr_result(stored)
        assert result.words == []
        assert result.error == "conversion failed"

    def test_non_string_error_is_stringified(self):
        # エンドポイント応答の error はそのまま保存されうる（文字列とは限らない）
        result = to_ocr_result({"error": {"code": 500}})
        assert result.words == []
        assert isinstance(result.error, str)

    @pytest.mark.parametrize("stored", [None, {}, {"words": None}, {"words": "abc"}, "broken"])
    def test_unusable_records_return_no_words(self, stored):
        result = to_ocr_result(stored)
        assert result.words == []
