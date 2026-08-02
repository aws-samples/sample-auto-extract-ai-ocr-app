"""Bedrock を呼んで応答を読み取る共通処理のテスト。

想定している正しい挙動:
- 読み取れたらその結果を返す。呼び出しは 1 回で終える。
- 読み取れなかったとき、モデルが応答を出し切っている（end_turn / stop_sequence）なら
  もう一度生成させれば通る見込みがあるので 1 回だけ再試行する。
- それ以外の停止理由（トークン上限など）は打ち切られた応答なので、再実行しても同じく
  打ち切られる見込みが高く、再試行せず即座に失敗させる。停止理由が読めない応答も
  「出し切った」と判断できないため再試行しない。
- 再試行しても読み取れなければ ResponseParseError を投げる。
- 読み取り以外の例外（API エラー等）は再試行せずそのまま伝える。
  API エラーの再試行は boto3 クライアントの retries 設定が担う。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

import clients.bedrock as bedrock_module
from clients.bedrock import call_bedrock_and_parse
from exceptions import ResponseParseError

MESSAGES = [{"role": "user", "content": [{"text": "抽出してください"}]}]
SYSTEM = [{"text": "あなたは抽出アシスタントです"}]


def _converse(text, stop_reason="end_turn"):
    response = {"output": {"message": {"content": [{"text": text}]}}}
    if stop_reason is not None:
        response["stopReason"] = stop_reason
    return response


class _Bedrock:
    """呼ばれた回数を数え、あらかじめ決めた応答を順に返す。"""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def __call__(self, messages, system_prompts, **kwargs):
        self.calls += 1
        index = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[index]


def _needs_ok(text):
    if text != "ok":
        raise ResponseParseError(f"unexpected response: {text}")
    return {"parsed": text}


@pytest.fixture
def bedrock(monkeypatch):
    def _install(responses):
        fake = _Bedrock(responses)
        monkeypatch.setattr(bedrock_module, "call_bedrock", fake)
        return fake
    return _install


class TestCallBedrockAndParse:
    def test_returns_parsed_result_without_retrying(self, bedrock):
        fake = bedrock([_converse("ok")])

        assert call_bedrock_and_parse(MESSAGES, SYSTEM, _needs_ok) == {"parsed": "ok"}
        assert fake.calls == 1

    @pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
    def test_retries_once_when_model_finished(self, bedrock, stop_reason):
        fake = bedrock([_converse("ng", stop_reason), _converse("ok", stop_reason)])

        assert call_bedrock_and_parse(MESSAGES, SYSTEM, _needs_ok) == {"parsed": "ok"}
        assert fake.calls == 2

    def test_raises_after_the_retry(self, bedrock):
        fake = bedrock([_converse("ng")])

        with pytest.raises(ResponseParseError):
            call_bedrock_and_parse(MESSAGES, SYSTEM, _needs_ok)
        assert fake.calls == 2

    @pytest.mark.parametrize("stop_reason", ["max_tokens", "content_filtered", None])
    def test_does_not_retry_for_other_stop_reasons(self, bedrock, stop_reason):
        fake = bedrock([_converse("ng", stop_reason)])

        with pytest.raises(ResponseParseError) as e:
            call_bedrock_and_parse(MESSAGES, SYSTEM, _needs_ok)
        assert fake.calls == 1
        # 原因を追えるよう停止理由をメッセージに残す
        assert str(stop_reason) in str(e.value)

    def test_api_errors_are_not_retried(self, monkeypatch):
        calls = []

        def _boom(messages, system_prompts, **kwargs):
            calls.append(1)
            raise RuntimeError("bedrock is down")

        monkeypatch.setattr(bedrock_module, "call_bedrock", _boom)

        with pytest.raises(RuntimeError):
            call_bedrock_and_parse(MESSAGES, SYSTEM, _needs_ok)
        assert len(calls) == 1

    def test_passes_model_overrides_through(self, monkeypatch):
        received = {}

        def _capture(messages, system_prompts, **kwargs):
            received.update(kwargs)
            return _converse("ok")

        monkeypatch.setattr(bedrock_module, "call_bedrock", _capture)

        call_bedrock_and_parse(MESSAGES, SYSTEM, _needs_ok, model_id="other-model")
        assert received == {"model_id": "other-model"}
