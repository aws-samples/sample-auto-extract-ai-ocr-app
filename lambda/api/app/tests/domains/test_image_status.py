import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from domains.image_status import (
    determine_parent_status,
    determine_parent_agent_status,
    ImageStatus, AgentStatus, PageProcessingMode,
    validate_image_status, validate_agent_status, validate_page_processing_mode,
)


class TestStatusValidators:
    def test_valid_image_status_passes(self):
        for s in ImageStatus:
            validate_image_status(s)  # 例外が出ないこと

    def test_invalid_image_status_raises(self):
        with pytest.raises(ValueError):
            validate_image_status("hoge")

    def test_not_started_is_not_a_valid_image_status(self):
        # "not_started" は read 時のデフォルト表示値であり write 値ではない
        assert "not_started" not in set(ImageStatus)
        with pytest.raises(ValueError):
            validate_image_status("not_started")

    def test_valid_agent_status_passes(self):
        for s in AgentStatus:
            validate_agent_status(s)

    def test_invalid_agent_status_raises(self):
        with pytest.raises(ValueError):
            validate_agent_status("bogus")

    def test_valid_page_processing_mode_passes(self):
        for m in PageProcessingMode:
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


class TestDetermineParentAgentStatus:
    """子ページの AI 検証状態から親の表示状態を決める。

    想定している正しい挙動: 検証が 1 つでも失敗していれば親は failed。
    まだ動いている / これから動く子がいる間は processing。全部終わったら
    completed（検証対象外の子が混じっていても、実行した子が完了なら完了扱い）。
    まだ誰も検証を始めていない状態は idle。
    """

    def test_no_children_is_idle(self):
        assert determine_parent_agent_status([]) == AgentStatus.IDLE

    def test_all_idle_is_idle(self):
        children = [{"agent_status": AgentStatus.IDLE}, {"agent_status": AgentStatus.IDLE}]
        assert determine_parent_agent_status(children) == AgentStatus.IDLE

    @pytest.mark.parametrize("missing", [{}, {"agent_status": None}, {"agent_status": ""}])
    def test_missing_agent_status_counts_as_idle(self, missing):
        assert determine_parent_agent_status([missing]) == AgentStatus.IDLE

    def test_any_failed_wins(self):
        children = [
            {"agent_status": AgentStatus.COMPLETED},
            {"agent_status": AgentStatus.FAILED},
            {"agent_status": AgentStatus.PROCESSING},
        ]
        assert determine_parent_agent_status(children) == AgentStatus.FAILED

    def test_any_processing_makes_parent_processing(self):
        children = [
            {"agent_status": AgentStatus.COMPLETED},
            {"agent_status": AgentStatus.PROCESSING},
        ]
        assert determine_parent_agent_status(children) == AgentStatus.PROCESSING

    def test_partially_idle_is_processing(self):
        # 一部だけ未実行 = 残りが処理される見込みなので処理中として見せる
        children = [
            {"agent_status": AgentStatus.COMPLETED},
            {"agent_status": AgentStatus.IDLE},
        ]
        assert determine_parent_agent_status(children) == AgentStatus.PROCESSING

    def test_all_completed(self):
        children = [{"agent_status": AgentStatus.COMPLETED}] * 2
        assert determine_parent_agent_status(children) == AgentStatus.COMPLETED

    def test_all_skipped(self):
        children = [{"agent_status": AgentStatus.SKIPPED}] * 2
        assert determine_parent_agent_status(children) == AgentStatus.SKIPPED

    def test_completed_and_skipped_mix_is_completed(self):
        # 抽出に失敗したページは skipped になる。成功ページと混在しても
        # 親は完了として落ち着く（processing のまま止まらない）
        children = [
            {"agent_status": AgentStatus.COMPLETED},
            {"agent_status": AgentStatus.SKIPPED},
        ]
        assert determine_parent_agent_status(children) == AgentStatus.COMPLETED
