"""抽出でパースに失敗したときのステータス遷移のテスト。

想定している正しい挙動:
- 抽出に成功したら extracting → completed。抽出値が保存される。
- 応答のパースに失敗したら completed にしてはいけない（失敗を完了と表示させない）。
  モデルが応答を出し切っている（stopReason=end_turn / stop_sequence）場合は
  再実行で直る見込みがあるので 1 回だけ再試行し、それでも駄目なら failed。
- トークン上限などそれ以外の停止理由は、同じ入力で再実行しても同じ結果になるため
  再試行せず即 failed にする（無駄な課金と待ち時間を避ける）。
- 失敗時は抽出値を保存しない。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json

import pytest

import services.extraction_service as extraction_module
from services.extraction_service import SingleImageExtractor
from domains.extraction_engine import ExtractionParseError
from domains.image_status import ImageStatus, AgentStatus

IMAGE_ID = "img-1"

FIELDS = {"fields": [{"name": "total", "display_name": "合計", "type": "string"}]}


def _converse(text, stop_reason="end_turn"):
    response = {"output": {"message": {"content": [{"text": text}]}}}
    if stop_reason is not None:
        response["stopReason"] = stop_reason
    return response


def _valid_body():
    return json.dumps({"extracted_data": {"total": "1000"}, "indices": {"total": [0]}})


class _Bedrock:
    """呼ばれた回数を数え、あらかじめ決めた応答を順に返す。"""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def __call__(self, messages, system_prompts):
        self.calls += 1
        # 応答が尽きたら最後のものを繰り返す
        index = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[index]


@pytest.fixture
def patched(monkeypatch):
    state = {
        "status_updates": [],
        "agent_updates": [],
        "extracted": [],
        "mappings": [],
        # 画像ステータスと検証ステータスの前後関係を見るための時系列
        "events": [],
        "agent_enabled": False,
    }

    monkeypatch.setattr(extraction_module.settings, "ENABLE_OCR", True)
    monkeypatch.setattr(
        extraction_module, "get_image",
        lambda image_id: {"app_name": "app", "converted_s3_key": ["k"]},
    )
    monkeypatch.setattr(extraction_module, "get_extraction_fields_for_app", lambda app: FIELDS)
    monkeypatch.setattr(extraction_module, "get_custom_prompt_for_app", lambda app: None)
    monkeypatch.setattr(
        extraction_module, "get_app_schema",
        lambda app: {"agent_enabled": state["agent_enabled"], "agent_auto_run": state["agent_enabled"]},
    )
    def _record_image_status(image_id, status):
        state["status_updates"].append(status)
        state["events"].append(("image", status))

    def _record_agent_status(image_id, status):
        state["agent_updates"].append(status)
        state["events"].append(("agent", status))

    monkeypatch.setattr(extraction_module, "update_image_status", _record_image_status)
    monkeypatch.setattr(extraction_module, "update_agent_status", _record_agent_status)

    def _record_extracted(image_id, info, mapping, extracted_fields=None):
        state["extracted"].append(info)
        state["mappings"].append(mapping)

    monkeypatch.setattr(extraction_module, "update_extracted_info", _record_extracted)
    return state


def _boom(*args, **kwargs):
    raise AssertionError("この経路の Bedrock 呼び出しは使われないはず")


def _extractor(monkeypatch, bedrock=None):
    """OCR 有効経路用の抽出器。bedrock 未指定なら _call_bedrock は呼ばれてはいけない。"""
    extractor = SingleImageExtractor(IMAGE_ID)
    monkeypatch.setattr(extractor, "_fetch_images", lambda image_data: (b"img", "image/jpeg"))
    monkeypatch.setattr(extractor, "_get_ocr_data", lambda image_data: {"words": []})
    monkeypatch.setattr(extractor, "_call_bedrock", bedrock or _boom)
    return extractor


class TestExtractionSuccess:
    def test_completes_and_saves_extracted_info(self, patched, monkeypatch):
        bedrock = _Bedrock([_converse(_valid_body())])
        _extractor(monkeypatch, bedrock).extract()

        assert patched["status_updates"] == [ImageStatus.EXTRACTING, ImageStatus.COMPLETED]
        assert patched["extracted"] == [{"total": "1000"}]
        assert bedrock.calls == 1

    def test_marks_agent_processing_before_completing_when_auto_run(self, patched, monkeypatch):
        # 検証中を先に立てないと、一覧で一瞬「完了」と出てから「検証中」に変わる
        patched["agent_enabled"] = True
        bedrock = _Bedrock([_converse(_valid_body())])
        _extractor(monkeypatch, bedrock).extract()

        assert patched["events"] == [
            ("image", ImageStatus.EXTRACTING),
            ("agent", AgentStatus.PROCESSING),
            ("image", ImageStatus.COMPLETED),
        ]

    def test_does_not_touch_agent_status_when_auto_run_is_off(self, patched, monkeypatch):
        bedrock = _Bedrock([_converse(_valid_body())])
        _extractor(monkeypatch, bedrock).extract()

        assert patched["agent_updates"] == []


class TestExtractionParseFailure:
    def test_retries_once_and_completes_when_second_attempt_parses(self, patched, monkeypatch):
        bedrock = _Bedrock([_converse("パースできない応答"), _converse(_valid_body())])
        _extractor(monkeypatch, bedrock).extract()

        assert bedrock.calls == 2
        assert patched["status_updates"] == [ImageStatus.EXTRACTING, ImageStatus.COMPLETED]
        assert patched["extracted"] == [{"total": "1000"}]

    def test_fails_after_retry_and_saves_nothing(self, patched, monkeypatch):
        bedrock = _Bedrock([_converse("パースできない応答")])
        with pytest.raises(ExtractionParseError):
            _extractor(monkeypatch, bedrock).extract()

        assert bedrock.calls == 2
        assert patched["status_updates"] == [ImageStatus.EXTRACTING, ImageStatus.FAILED]
        assert patched["extracted"] == []

    @pytest.mark.parametrize("stop_reason", ["max_tokens", "content_filtered"])
    def test_does_not_retry_when_model_stopped_for_other_reasons(
        self, patched, monkeypatch, stop_reason
    ):
        bedrock = _Bedrock([_converse("途中で切れた応答", stop_reason=stop_reason)])
        with pytest.raises(ExtractionParseError):
            _extractor(monkeypatch, bedrock).extract()

        assert bedrock.calls == 1
        assert patched["status_updates"] == [ImageStatus.EXTRACTING, ImageStatus.FAILED]
        assert patched["extracted"] == []

    def test_does_not_retry_when_stop_reason_is_absent(self, patched, monkeypatch):
        # 停止理由が読めない応答は「出し切った」と判断できないので再試行しない
        bedrock = _Bedrock([_converse("パースできない応答", stop_reason=None)])
        with pytest.raises(ExtractionParseError):
            _extractor(monkeypatch, bedrock).extract()

        assert bedrock.calls == 1
        assert patched["status_updates"] == [ImageStatus.EXTRACTING, ImageStatus.FAILED]

    @pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
    def test_retries_for_every_completed_stop_reason(self, patched, monkeypatch, stop_reason):
        bedrock = _Bedrock([
            _converse("パースできない応答", stop_reason=stop_reason),
            _converse(_valid_body(), stop_reason=stop_reason),
        ])
        _extractor(monkeypatch, bedrock).extract()

        assert bedrock.calls == 2
        assert patched["status_updates"][-1] == ImageStatus.COMPLETED


class TestExtractionWithoutOcr:
    """OCR 無効時は OCR 語との対応表を作らず、応答から JSON だけを取り出す。

    この経路はクラスの `_call_bedrock` ではなくモジュール関数 `call_bedrock` を使うため、
    `_extractor` 側の `_call_bedrock` は呼ばれたら失敗する sentinel にしてある。
    """

    def test_fails_when_no_json_in_response(self, patched, monkeypatch):
        monkeypatch.setattr(extraction_module.settings, "ENABLE_OCR", False)
        bedrock = _Bedrock([_converse("JSONを含まない応答")])
        monkeypatch.setattr(extraction_module, "call_bedrock", bedrock)

        with pytest.raises(ExtractionParseError):
            _extractor(monkeypatch).extract()

        assert bedrock.calls == 2
        assert patched["status_updates"] == [ImageStatus.EXTRACTING, ImageStatus.FAILED]
        assert patched["extracted"] == []

    def test_completes_when_json_present(self, patched, monkeypatch):
        monkeypatch.setattr(extraction_module.settings, "ENABLE_OCR", False)
        bedrock = _Bedrock([_converse('{"total": "1000"}')])
        monkeypatch.setattr(extraction_module, "call_bedrock", bedrock)

        _extractor(monkeypatch).extract()

        assert bedrock.calls == 1
        assert patched["status_updates"] == [ImageStatus.EXTRACTING, ImageStatus.COMPLETED]
        assert patched["extracted"] == [{"total": "1000"}]
        # 対応表は作らない
        assert patched["mappings"] == [{}]
