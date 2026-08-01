import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import asyncio
import json

import pytest

import services.ocr_service as ocr_module
from services.ocr_service import OcrService
from domains.image_status import ImageStatus, AgentStatus
from exceptions import BadRequestError, EndpointNotReadyError


class _FakeSfn:
    def __init__(self):
        self.calls = []

    def start_execution(self, **kwargs):
        self.calls.append(kwargs)
        return {"executionArn": "arn:aws:states:::execution/test"}


@pytest.fixture
def patched(monkeypatch):
    """外部依存を差し替え、呼び出しを記録する。"""
    state = {
        "status_updates": [],   # (image_id, status)
        "agent_updates": [],    # (image_id, status)
        "images": {},           # image_id -> record (get_image 用)
        "app_images": [],       # get_images(app_name) の戻り
    }

    fake_sfn = _FakeSfn()
    monkeypatch.setattr(ocr_module, "sfn_client", fake_sfn)
    monkeypatch.setattr(ocr_module.settings, "STATE_MACHINE_ARN", "arn:sm")
    monkeypatch.setattr(
        ocr_module, "update_image_status",
        lambda image_id, status, job_id=None: state["status_updates"].append((image_id, status)),
    )
    monkeypatch.setattr(
        ocr_module, "update_agent_status",
        lambda image_id, status, suggestions_count=None: state["agent_updates"].append((image_id, status)),
    )
    monkeypatch.setattr(ocr_module, "get_image", lambda image_id: state["images"].get(image_id))
    monkeypatch.setattr(ocr_module, "get_images", lambda app_name: state["app_images"])

    state["sfn"] = fake_sfn
    return state


def _make_service(monkeypatch, enable_ocr=True, endpoint_ready=True):
    monkeypatch.setattr(ocr_module.settings, "ENABLE_OCR", enable_ocr)
    svc = OcrService()
    monkeypatch.setattr(svc, "get_endpoint_status", lambda: {"ready": endpoint_ready})
    return svc


class TestStartPipeline:
    def test_sets_ocr_status_and_agent_reset_and_starts_one_execution(self, patched, monkeypatch):
        svc = _make_service(monkeypatch)
        patched["images"] = {"a": {"status": "pending"}, "b": {"status": "pending"}}

        result = asyncio.run(svc._start_pipeline(["a", "b"], skip_ocr=False))

        assert "jobId" in result
        # 各画像が OCR に、agent が idle にリセットされる
        assert ("a", ImageStatus.OCR) in patched["status_updates"]
        assert ("b", ImageStatus.OCR) in patched["status_updates"]
        assert ("a", AgentStatus.IDLE) in patched["agent_updates"]
        # SFn 実行は 1 回、images に全 id が入る
        assert len(patched["sfn"].calls) == 1
        payload = json.loads(patched["sfn"].calls[0]["input"])
        assert [i["image_id"] for i in payload["images"]] == ["a", "b"]

    def test_dedupes_ids(self, patched, monkeypatch):
        svc = _make_service(monkeypatch)
        patched["images"] = {"a": {"status": "pending"}}
        asyncio.run(svc._start_pipeline(["a", "a", "a"], skip_ocr=False))
        payload = json.loads(patched["sfn"].calls[0]["input"])
        assert [i["image_id"] for i in payload["images"]] == ["a"]

    def test_empty_list_is_noop(self, patched, monkeypatch):
        svc = _make_service(monkeypatch)
        result = asyncio.run(svc._start_pipeline([], skip_ocr=False))
        assert "jobId" in result
        assert len(patched["sfn"].calls) == 0
        assert patched["status_updates"] == []

    def test_endpoint_not_ready_raises_before_status_change(self, patched, monkeypatch):
        svc = _make_service(monkeypatch, endpoint_ready=False)
        monkeypatch.setattr(ocr_module, "trigger_endpoint_wakeup", lambda *a, **k: None)
        monkeypatch.setattr(ocr_module.settings, "OCR_ENGINE", "paddle")

        with pytest.raises(EndpointNotReadyError):
            asyncio.run(svc._start_pipeline(["a"], skip_ocr=False))
        # ステータスは一切触られていない
        assert patched["status_updates"] == []
        assert len(patched["sfn"].calls) == 0

    def test_rollback_restores_prior_status(self, patched, monkeypatch):
        svc = _make_service(monkeypatch)
        # 元 COMPLETED の画像を再処理し、SFn 起動が失敗するケース
        patched["images"] = {"a": {"status": "completed"}}

        def boom(**kwargs):
            raise RuntimeError("sfn down")
        monkeypatch.setattr(patched["sfn"], "start_execution", boom)

        with pytest.raises(RuntimeError):
            asyncio.run(svc._start_pipeline(["a"], skip_ocr=False))

        # OCR にした後、prior の completed に戻す（pending 固定にしない）
        assert ("a", ImageStatus.OCR) in patched["status_updates"]
        assert ("a", "completed") in patched["status_updates"]


class TestBatchResolver:
    def test_omitted_ids_selects_pending_excluding_parent_container(self, patched, monkeypatch):
        svc = _make_service(monkeypatch)
        # p1 は個別ページの親コンテナ（子 c1 が p1 を parent に持つ）→ 除外
        patched["app_images"] = [
            {"id": "p1", "status": "pending"},
            {"id": "c1", "status": "pending", "parent_document_id": "p1"},
            {"id": "s1", "status": "pending"},
            {"id": "done", "status": "completed"},
        ]
        patched["images"] = {img["id"]: img for img in patched["app_images"]}

        class Req:
            app_name = "app"
            image_ids = None
            skip_ocr = False

        asyncio.run(svc.start_step_functions_job(Req()))
        payload = json.loads(patched["sfn"].calls[0]["input"])
        started = sorted(i["image_id"] for i in payload["images"])
        # 親コンテナ p1 と completed は除外、c1 と s1 のみ
        assert started == ["c1", "s1"]

    def test_provided_ids_rejects_cross_app(self, patched, monkeypatch):
        svc = _make_service(monkeypatch)
        patched["app_images"] = [{"id": "a", "status": "pending"}]

        class Req:
            app_name = "app"
            image_ids = ["a", "foreign"]
            skip_ocr = False

        with pytest.raises(BadRequestError):
            asyncio.run(svc.start_step_functions_job(Req()))
        assert len(patched["sfn"].calls) == 0

    def test_provided_ids_starts_only_those(self, patched, monkeypatch):
        svc = _make_service(monkeypatch)
        patched["app_images"] = [
            {"id": "a", "status": "pending"},
            {"id": "b", "status": "pending"},
        ]
        patched["images"] = {img["id"]: img for img in patched["app_images"]}

        class Req:
            app_name = "app"
            image_ids = ["a"]
            skip_ocr = False

        asyncio.run(svc.start_step_functions_job(Req()))
        payload = json.loads(patched["sfn"].calls[0]["input"])
        assert [i["image_id"] for i in payload["images"]] == ["a"]


class TestSingleWrapper:
    def test_delegates_to_pipeline(self, patched, monkeypatch):
        svc = _make_service(monkeypatch)
        patched["images"] = {"x": {"status": "completed"}}
        result = asyncio.run(svc.start_step_functions_for_image("x", skip_ocr=True))
        assert "jobId" in result
        payload = json.loads(patched["sfn"].calls[0]["input"])
        assert payload["images"] == [{"image_id": "x", "skip_ocr": True}]
