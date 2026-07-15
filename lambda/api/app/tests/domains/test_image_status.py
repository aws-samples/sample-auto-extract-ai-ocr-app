import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from domains.image_status import to_api_status, determine_parent_status


class TestToApiStatus:
    @pytest.mark.parametrize("internal", ["ocr", "extracting"])
    def test_in_progress_phases_fold_to_processing(self, internal):
        assert to_api_status(internal) == "processing"

    @pytest.mark.parametrize(
        "value",
        ["uploading", "pending", "converting", "processing", "completed", "failed"],
    )
    def test_other_values_are_identity(self, value):
        assert to_api_status(value) == value

    def test_none_stays_none(self):
        assert to_api_status(None) is None


class TestDetermineParentStatus:
    def test_empty_children_is_converting(self):
        assert determine_parent_status([]) == "converting"

    def test_all_completed(self):
        children = [{"status": "completed"}, {"status": "completed"}]
        assert determine_parent_status(children) == "completed"

    def test_any_failed_wins_over_completed(self):
        children = [{"status": "completed"}, {"status": "failed"}]
        assert determine_parent_status(children) == "failed"

    @pytest.mark.parametrize("phase", ["ocr", "extracting", "processing"])
    def test_in_progress_child_makes_parent_processing(self, phase):
        # ocr / extracting の子も processing として集約する
        children = [{"status": phase}, {"status": "pending"}]
        assert determine_parent_status(children) == "processing"

    def test_no_progress_no_terminal_is_converting(self):
        children = [{"status": "pending"}, {"status": "pending"}]
        assert determine_parent_status(children) == "converting"

    @pytest.mark.parametrize("phase", ["ocr", "extracting"])
    def test_parent_never_stores_raw_internal_phase(self, phase):
        # 親に内部フェーズが保存されるとフロントに漏れるため processing に丸める
        children = [{"status": phase}]
        assert determine_parent_status(children) not in ("ocr", "extracting")


class TestImageInfoStatusFold:
    """一覧レスポンスの境界（ImageInfo）で内部フェーズが畳まれることを保証する。"""

    @pytest.mark.parametrize(
        "internal,expected",
        [("ocr", "processing"), ("extracting", "processing"),
         ("completed", "completed"), ("failed", "failed")],
    )
    def test_serialized_status_is_folded(self, internal, expected):
        from schemas.image import ImageInfo
        info = ImageInfo.model_validate({"id": "x", "status": internal})
        assert info.model_dump(by_alias=True)["status"] == expected
