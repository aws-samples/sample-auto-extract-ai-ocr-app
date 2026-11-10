import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from domains.ocr_engine import (
    _box_to_points,
    _normalize_points,
    parse_llm_ocr_response,
    parse_ocr_response,
    parse_yomitoku_mp_response,
)
from exceptions import ResponseParseError


class TestNormalizePoints:
    def test_floats_are_rounded_to_int(self):
        points = [[1.4, 2.6], [3.5, 4.0]]
        assert _normalize_points(points) == [[1, 3], [4, 4]]
        assert all(isinstance(v, int) for p in _normalize_points(points) for v in p)

    def test_ints_are_unchanged(self):
        points = [[1, 2], [3, 4]]
        assert _normalize_points(points) == [[1, 2], [3, 4]]

    def test_four_point_structure_preserved(self):
        points = [[0.1, 0.9], [10.2, 0.9], [10.2, 5.1], [0.1, 5.1]]
        result = _normalize_points(points)
        assert len(result) == 4
        assert result == [[0, 1], [10, 1], [10, 5], [0, 5]]

    def test_none_passthrough(self):
        assert _normalize_points(None) is None

    def test_non_list_passthrough(self):
        assert _normalize_points("not-a-list") == "not-a-list"

    def test_non_list_of_list_passthrough(self):
        # 座標が list でない要素は素通し
        assert _normalize_points([1, 2, 3]) == [1, 2, 3]

    def test_non_numeric_coordinate_passthrough(self):
        # 数値でない座標値は変換せずそのまま
        assert _normalize_points([["a", "b"]]) == [["a", "b"]]

    def test_bool_not_treated_as_number(self):
        # bool は数値に丸めない
        assert _normalize_points([[True, False]]) == [[True, False]]


class TestParseOcrResponse:
    def test_points_normalized_to_int_and_rec_score_dropped(self):
        response = {
            "words": [
                {"id": 0, "content": "abc", "rec_score": 0.98,
                 "points": [[1.4, 2.6], [3.5, 4.0]]},
            ]
        }
        result = parse_ocr_response(response)
        word = result["words"][0]
        assert word["points"] == [[1, 3], [4, 4]]
        assert "rec_score" not in word
        assert result["word_count"] == 1
        assert result["text"] == "abc"

    def test_error_response_passthrough(self):
        response = {"error": "boom"}
        assert parse_ocr_response(response) == {"error": "boom"}


class TestBoxToPoints:
    def test_box_converted_to_four_corner_points(self):
        # [x_min, y_min, x_max, y_max] → 左上から時計回りの四隅
        points = _box_to_points([10, 20, 100, 50], 200, 100)
        assert points == [[10, 20], [100, 20], [100, 50], [10, 50]]

    def test_coordinates_clamped_to_image_bounds(self):
        points = _box_to_points([-5, -5, 250, 150], 200, 100)
        assert points == [[0, 0], [200, 0], [200, 100], [0, 100]]

    def test_float_coordinates_rounded_to_int(self):
        points = _box_to_points([10.4, 20.6, 99.5, 49.9], 200, 100)
        assert points == [[10, 21], [100, 21], [100, 50], [10, 50]]

    def test_invalid_box_returns_none(self):
        assert _box_to_points(None, 200, 100) is None
        assert _box_to_points([1, 2, 3], 200, 100) is None
        assert _box_to_points([1, 2, 3, "x"], 200, 100) is None
        assert _box_to_points("not-a-list", 200, 100) is None

    def test_non_finite_coordinates_return_none(self):
        # json.loads は NaN/Infinity リテラルを受理するため、到達しうる入力
        assert _box_to_points([float("nan"), 0, 10, 10], 200, 100) is None
        assert _box_to_points([0, 0, float("inf"), 10], 200, 100) is None

    def test_inverted_or_degenerate_box_returns_none(self):
        # min/max が反転した box や面積ゼロの box は points にしない
        assert _box_to_points([100, 50, 10, 20], 200, 100) is None
        assert _box_to_points([10, 10, 10, 50], 200, 100) is None
        # 画像範囲外のみに存在する box もクランプで面積ゼロになり None
        assert _box_to_points([300, 10, 400, 50], 200, 100) is None


class TestParseLlmOcrResponse:
    """LLM OCR 応答のパース。

    想定している正しい挙動: {"words": [{"text", "box"}]} を既存 OCR 契約
    （{"id", "content", "points"}）に変換する。box が不正な word は points なしで
    含め、words リスト自体が取れない応答は ResponseParseError を投げる。
    """

    def test_parses_words_with_points(self):
        text = '{"words": [{"text": "請求書", "box": [10, 20, 100, 50]}, {"text": "合計", "box": [10, 60, 60, 80]}]}'
        result = parse_llm_ocr_response(text, 200, 100)

        assert result["word_count"] == 2
        assert result["words"][0] == {
            "id": 0, "content": "請求書",
            "points": [[10, 20], [100, 20], [100, 50], [10, 50]],
        }
        assert result["words"][1]["id"] == 1
        assert result["text"] == "請求書 合計"

    def test_markdown_fenced_json_is_parsed(self):
        text = '```json\n{"words": [{"text": "行1", "box": [0, 0, 10, 10]}]}\n```'
        result = parse_llm_ocr_response(text, 200, 100)
        assert result["word_count"] == 1

    def test_word_with_invalid_box_kept_without_points(self):
        text = '{"words": [{"text": "行1", "box": "broken"}]}'
        result = parse_llm_ocr_response(text, 200, 100)
        assert result["words"] == [{"id": 0, "content": "行1"}]

    def test_empty_text_words_are_skipped(self):
        text = '{"words": [{"text": "  ", "box": [0, 0, 1, 1]}, {"text": "a", "box": [0, 0, 1, 1]}]}'
        result = parse_llm_ocr_response(text, 200, 100)
        assert result["word_count"] == 1
        assert result["words"][0]["content"] == "a"

    def test_empty_words_returns_empty_result(self):
        result = parse_llm_ocr_response('{"words": []}', 200, 100)
        assert result == {"text": "", "words": [], "word_count": 0}

    def test_non_json_raises(self):
        with pytest.raises(ResponseParseError):
            parse_llm_ocr_response("行1のテキスト\n行2のテキスト", 200, 100)

    def test_json_without_words_list_raises(self):
        with pytest.raises(ResponseParseError):
            parse_llm_ocr_response('{"lines": ["a"]}', 200, 100)


class TestParseYomitokuResponse:
    def test_points_normalized_to_int(self):
        response = {
            "result": [
                {"words": [
                    {"content": "xyz", "points": [[5.7, 6.2]], "direction": "horizontal"},
                ]}
            ]
        }
        result = parse_yomitoku_mp_response(response)
        word = result["words"][0]
        assert word["points"] == [[6, 6]]
        assert word["content"] == "xyz"
        assert result["word_count"] == 1
