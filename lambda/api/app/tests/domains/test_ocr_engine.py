import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from domains.ocr_engine import (
    _normalize_points,
    parse_ocr_response,
    parse_yomitoku_mp_response,
)


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
