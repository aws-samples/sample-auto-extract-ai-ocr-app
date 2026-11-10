"""複数ページ OCR 結果の統合保存のテスト。

想定している正しい挙動: 全ページ通してユニークな ID を word に振り直し、
ページ別結果（pages）にも同じ ID が反映される。points を持たない word や
同一テキストの行が複数あっても、ID の重複・欠落は起きない。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

import services.ocr_service as ocr_module
from services.ocr_service import OcrService


@pytest.fixture
def saved(monkeypatch):
    state = {}
    monkeypatch.setattr(
        ocr_module, "db_update_ocr_result",
        lambda image_id, result: state.update({"image_id": image_id, "result": result}),
    )
    monkeypatch.setattr(ocr_module, "sync_parent_status", lambda image_id: None)
    return state


class TestSaveMultipageOcrResult:
    def test_assigns_globally_unique_ids(self, saved):
        ocr_results = [
            {"page": 1, "words": [{"content": "a", "points": [[0, 0]]}], "text": "a"},
            {"page": 2, "words": [{"content": "b", "points": [[1, 1]]}], "text": "b"},
        ]

        OcrService._save_multipage_ocr_result("img-1", ocr_results)

        result = saved["result"]
        assert [w["id"] for w in result["words"]] == [0, 1]
        assert result["total_pages"] == 2
        # ページ別結果にも同じ ID / page が反映される
        assert result["pages"][0]["words"][0]["id"] == 0
        assert result["pages"][1]["words"][0]["id"] == 1
        assert result["pages"][1]["words"][0]["page"] == 2

    def test_duplicate_content_without_points_keeps_distinct_ids(self, saved):
        # LLM OCR のフォールバック等で points を持たない同一テキスト行が並ぶケース
        ocr_results = [
            {"page": 1, "words": [{"content": "合計"}, {"content": "合計"}], "text": "合計 合計"},
        ]

        OcrService._save_multipage_ocr_result("img-1", ocr_results)

        page_words = saved["result"]["pages"][0]["words"]
        assert [w["id"] for w in page_words] == [0, 1]
