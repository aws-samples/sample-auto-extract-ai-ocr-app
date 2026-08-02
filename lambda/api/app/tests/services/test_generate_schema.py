"""帳票からスキーマを自動生成する処理のテスト。

想定している正しい挙動: 抽出と同じく、応答を読み取れなかったら 1 回だけ再試行してから
失敗させる。読み取れた応答はそのまま返す。

抽出処理とこの処理は同じ共通ヘルパーを通るため、リトライの挙動が片方だけ変わることはない。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import asyncio
import json

import pytest

import clients.bedrock as bedrock_module
import services.schema_service as schema_module
from services.schema_service import SchemaService
from exceptions import ResponseParseError

FIELDS = [{"name": "total", "display_name": "合計", "type": "string"}]


class _Request:
    s3_key = "uploads/sample.png"
    filename = "sample.png"
    instructions = None


def _converse(text, stop_reason="end_turn"):
    return {
        "stopReason": stop_reason,
        "output": {"message": {"content": [{"text": text}]}},
    }


class _Bedrock:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def __call__(self, messages, system_prompts, **kwargs):
        self.calls += 1
        index = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[index]


class _Body:
    @staticmethod
    def read():
        return b"image-bytes"


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(
        schema_module, "s3_client",
        type("S3", (), {"get_object": staticmethod(lambda **kwargs: {"Body": _Body()})})(),
    )
    monkeypatch.setattr(
        schema_module, "build_schema_generation_request",
        lambda file_data, instructions: ([], []),
    )
    return SchemaService()


def _generate(service):
    return asyncio.run(service.generate_schema(_Request()))


class TestSchemaGenerationRetry:
    def test_returns_schema_without_retrying(self, service, monkeypatch):
        bedrock = _Bedrock([_converse(json.dumps({"fields": FIELDS}))])
        monkeypatch.setattr(bedrock_module, "call_bedrock", bedrock)

        assert _generate(service) == {"fields": FIELDS}
        assert bedrock.calls == 1

    def test_retries_once_when_response_is_not_readable(self, service, monkeypatch):
        bedrock = _Bedrock([
            _converse("スキーマを作れませんでした"),
            _converse(json.dumps({"fields": FIELDS})),
        ])
        monkeypatch.setattr(bedrock_module, "call_bedrock", bedrock)

        assert _generate(service) == {"fields": FIELDS}
        assert bedrock.calls == 2

    def test_fails_after_the_retry(self, service, monkeypatch):
        bedrock = _Bedrock([_converse("スキーマを作れませんでした")])
        monkeypatch.setattr(bedrock_module, "call_bedrock", bedrock)

        with pytest.raises(ResponseParseError):
            _generate(service)
        assert bedrock.calls == 2
