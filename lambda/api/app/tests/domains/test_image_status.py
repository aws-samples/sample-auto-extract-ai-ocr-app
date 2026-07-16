import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from domains.image_status import (
    determine_parent_status,
    ImageStatus, AgentStatus, PageProcessingMode,
    validate_image_status, validate_agent_status, validate_page_processing_mode,
)


class TestStatusValidators:
    def test_valid_image_status_passes(self):
        for s in ImageStatus.ALL:
            validate_image_status(s)  # 例外が出ないこと

    def test_invalid_image_status_raises(self):
        with pytest.raises(ValueError):
            validate_image_status("hoge")

    def test_not_started_is_not_a_valid_image_status(self):
        # "not_started" は read 時のデフォルト表示値であり write 値ではない
        assert "not_started" not in ImageStatus.ALL
        with pytest.raises(ValueError):
            validate_image_status("not_started")

    def test_valid_agent_status_passes(self):
        for s in AgentStatus.ALL:
            validate_agent_status(s)

    def test_invalid_agent_status_raises(self):
        with pytest.raises(ValueError):
            validate_agent_status("bogus")

    def test_valid_page_processing_mode_passes(self):
        for m in PageProcessingMode.ALL:
            validate_page_processing_mode(m)

    def test_invalid_page_processing_mode_raises(self):
        with pytest.raises(ValueError):
            validate_page_processing_mode("batch")


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
        # ocr / extracting / processing の子はまとめて処理中として集約する
        children = [{"status": phase}, {"status": "pending"}]
        assert determine_parent_status(children) == "processing"

    def test_no_progress_no_terminal_is_converting(self):
        children = [{"status": "pending"}, {"status": "pending"}]
        assert determine_parent_status(children) == "converting"
