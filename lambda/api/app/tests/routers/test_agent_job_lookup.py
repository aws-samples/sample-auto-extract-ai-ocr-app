"""画像の最新 AI 検証ジョブ取得のテスト。

想定している正しい挙動:
- 検証を実行していない状態（未設定 / idle）では過去のジョブを返さない。
  抽出をやり直すと agent_status は idle に戻るため、返してしまうと前回の修正提案が
  画面に復活する。
- 検証を実行した状態（processing / completed / failed）ならジョブを返す。
- 画像側が processing なのにジョブがまだ無い / ジョブ側が追いついていない場合は、
  完了ではなく processing として返す（検証中の表示を保つ）。
- 提案のうち未対応のものだけを返す。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import asyncio

import pytest

import routers.images as images_router
from routers.images import get_agent_job_by_image
from domains.image_status import AgentStatus


def _call(image_id="img-1"):
    return asyncio.run(get_agent_job_by_image(image_id, user={"id": 1, "role": "admin"}))


@pytest.fixture
def env(monkeypatch):
    state = {"image": {}, "job": None}
    monkeypatch.setattr(images_router, "get_image", lambda image_id: state["image"])
    monkeypatch.setattr(
        images_router, "get_latest_agent_job_by_image_id", lambda image_id: state["job"]
    )
    return state


PAST_JOB = {
    "id": "job-1",
    "status": "completed",
    "suggestions": [{"field": "total", "value": "1000"}],
}


class TestNotVerifiedStates:
    @pytest.mark.parametrize("agent_status", [None, AgentStatus.IDLE])
    def test_does_not_return_past_job(self, env, agent_status):
        env["image"] = {"agent_status": agent_status} if agent_status else {}
        env["job"] = PAST_JOB

        assert _call() == {"status": "none", "suggestions": []}


class TestVerifiedStates:
    def test_returns_completed_job_with_pending_suggestions(self, env):
        env["image"] = {"agent_status": AgentStatus.COMPLETED}
        env["job"] = PAST_JOB

        result = _call()
        assert result["status"] == "completed"
        assert result["job_id"] == "job-1"
        assert result["total_suggestions_count"] == 1

    def test_excludes_resolved_suggestions(self, env):
        env["image"] = {"agent_status": AgentStatus.COMPLETED}
        env["job"] = {
            "id": "job-1",
            "status": "completed",
            "suggestions": [
                {"field": "a", "status": "accepted"},
                {"field": "b", "status": "pending"},
            ],
        }

        result = _call()
        assert [s["field"] for s in result["suggestions"]] == ["b"]
        assert result["total_suggestions_count"] == 2
        # index は保存されている全提案の中での位置。採否の更新はこの位置を使って
        # 該当要素だけを書き換えるため、未対応分だけで数え直すと別の提案を壊す。
        assert result["suggestions"][0]["index"] == 1

    def test_returns_failed_job(self, env):
        env["image"] = {"agent_status": AgentStatus.FAILED}
        env["job"] = {"id": "job-1", "status": "failed", "error": "runtime error"}

        result = _call()
        assert result["status"] == "failed"
        assert result["error"] == "runtime error"

    def test_no_job_yet_returns_none(self, env):
        env["image"] = {"agent_status": AgentStatus.COMPLETED}
        env["job"] = None

        assert _call() == {"status": "none", "suggestions": []}

    def test_processing_without_job_stays_processing(self, env):
        env["image"] = {"agent_status": AgentStatus.PROCESSING}
        env["job"] = None

        assert _call() == {"status": "processing", "suggestions": []}

    def test_processing_image_wins_over_stale_job(self, env):
        # 前回の完了ジョブしか無い時点で「完了」と返すと、検証中の表示が消える
        env["image"] = {"agent_status": AgentStatus.PROCESSING}
        env["job"] = PAST_JOB

        assert _call() == {"status": "processing", "suggestions": []}
