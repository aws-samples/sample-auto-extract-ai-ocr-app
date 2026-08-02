"""AI 検証を実行せずに終わらせる条件のテスト。

想定している正しい挙動:
- OCR や抽出が失敗した画像は検証する中身が無いので、自動実行では検証しない。
  このとき agent_status は skipped にする（idle にすると、同じ PDF の他ページが
  検証済みでも親が「処理中」のまま止まる）。
- 手動実行はユーザーが明示的に指示しているので、失敗した画像でも実行する。
- ユースケースが検証を自動実行しない設定なら idle にする。設定を変えれば実行されうるので
  「まだ検証していない」状態として扱う。
- 一方、画像がどのユースケースにも紐付いていない場合は何を検証すべきか決まらないので
  skipped にする（設定を変えても実行されない）。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

import workers.agent_kick as agent_kick
from workers.agent_kick import agent_kick_handler
from domains.image_status import AgentStatus
from repositories.job_repository import JobStatus

IMAGE_ID = "img-1"


@pytest.fixture
def env(monkeypatch):
    state = {
        "image": {"status": "completed", "app_name": "app"},
        "schema": {"agent_enabled": True, "agent_auto_run": True},
        "agent_updates": [],
        "parent_syncs": [],
        "created_jobs": [],
        "job_updates": [],
    }

    monkeypatch.setattr(agent_kick, "get_image", lambda image_id: state["image"])
    monkeypatch.setattr(agent_kick, "get_app_schema", lambda app_name: state["schema"])
    monkeypatch.setattr(
        agent_kick, "update_agent_status",
        lambda image_id, status, **kwargs: state["agent_updates"].append(status),
    )
    monkeypatch.setattr(
        agent_kick, "sync_parent_agent_status",
        lambda image_id: state["parent_syncs"].append(image_id),
    )
    monkeypatch.setattr(
        agent_kick, "update_agent_job",
        lambda job_id, status, **kwargs: state["job_updates"].append((job_id, status)),
    )

    # 検証に進んだかどうかはジョブが作られたかで判定する。実際の検証（AgentCore 呼び出し）
    # には進ませたくないので、ジョブ作成を記録した上で以降を失敗させる。
    def _create_job(image_id):
        state["created_jobs"].append(image_id)
        raise RuntimeError("stop before invoking the agent")

    monkeypatch.setattr(agent_kick, "create_agent_job", _create_job)
    return state


class TestAgentKickSkip:
    def test_failed_image_is_skipped_and_parent_synced(self, env):
        env["image"] = {"status": "failed", "app_name": "app"}

        result = agent_kick_handler({"image_id": IMAGE_ID}, None)

        assert result["status"] == "skipped"
        assert env["agent_updates"] == [AgentStatus.SKIPPED]
        assert env["parent_syncs"] == [IMAGE_ID]
        assert env["created_jobs"] == []

    def test_manual_run_is_not_skipped_for_failed_image(self, env):
        env["image"] = {"status": "failed", "app_name": "app"}

        # 手動実行はガードを通り抜けて検証に進む
        agent_kick_handler({"image_id": IMAGE_ID, "manual": True}, None)

        assert env["created_jobs"] == [IMAGE_ID]
        assert AgentStatus.SKIPPED not in env["agent_updates"]

    def test_usecase_without_agent_is_idle_not_skipped(self, env):
        env["schema"] = {"agent_enabled": False, "agent_auto_run": False}

        result = agent_kick_handler({"image_id": IMAGE_ID}, None)

        assert result["status"] == "skipped"
        assert env["agent_updates"] == [AgentStatus.IDLE]

    def test_image_without_usecase_is_skipped(self, env):
        env["image"] = {"status": "completed"}

        result = agent_kick_handler({"image_id": IMAGE_ID}, None)

        assert result["status"] == "skipped"
        assert env["agent_updates"] == [AgentStatus.SKIPPED]


class TestSkippedJobFinalization:
    """スキップ時に検証ジョブを閉じるのは手動実行だけ。

    想定している正しい挙動: 手動実行はジョブを先に作ってから呼ぶので、検証せずに
    終わるならジョブを skipped にして画面のポーリングを止める。自動実行は job_id が
    渡ってこないため、渡された値を触ってジョブでないレコードを作らない。
    """

    def test_manual_run_closes_the_job(self, env):
        env["schema"] = {"agent_enabled": False, "agent_auto_run": False}

        agent_kick_handler({"image_id": IMAGE_ID, "manual": True, "job_id": "job-1"}, None)

        assert env["job_updates"] == [("job-1", JobStatus.SKIPPED)]

    def test_automatic_run_does_not_touch_any_job(self, env):
        env["schema"] = {"agent_enabled": False, "agent_auto_run": False}

        agent_kick_handler({"image_id": IMAGE_ID, "job_id": "not-an-agent-job"}, None)

        assert env["job_updates"] == []
