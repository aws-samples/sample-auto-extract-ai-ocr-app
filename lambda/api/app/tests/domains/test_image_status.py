import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from domains.image_status import determine_parent_status


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
