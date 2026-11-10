"""resolve_bedrock_ocr_model のテスト。

想定している正しい挙動:
- 空のモデル ID は既定の us.openai.gpt-6-astra になる。
- 許可リストのモデル（us. / global. の Astra 推論プロファイル）は、呼び出せるリージョンなら
  そのまま返す。
- 許可リスト外のモデル（他社モデル、推論プロファイルなしの openai.gpt-6-astra 等）は
  UnsupportedOcrModelError。
- 許可リストのモデルでも、呼び出せないリージョン（us. を東京から等）は UnsupportedOcrModelError。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from domains.ocr_engine import resolve_bedrock_ocr_model
from exceptions import UnsupportedOcrModelError


def test_empty_model_defaults_to_astra():
    assert resolve_bedrock_ocr_model("", "us-east-1") == ("us.openai.gpt-6-astra", "us-east-1")


@pytest.mark.parametrize("model_id, region", [
    ("us.openai.gpt-6-astra", "us-west-2"),
    ("us.openai.gpt-6-astra", "ca-central-1"),
    ("global.openai.gpt-6-astra", "ap-northeast-1"),
    ("global.openai.gpt-6-astra", "us-east-1"),
])
def test_allowed_model_and_region(model_id, region):
    assert resolve_bedrock_ocr_model(model_id, region) == (model_id, region)


@pytest.mark.parametrize("model_id", [
    "us.anthropic.claude-sonnet-4-6",
    "openai.gpt-6-astra",
    "us.openai.gpt-6-sol",
])
def test_unsupported_model_is_rejected(model_id):
    with pytest.raises(UnsupportedOcrModelError):
        resolve_bedrock_ocr_model(model_id, "us-east-1")


def test_us_profile_from_tokyo_is_rejected():
    with pytest.raises(UnsupportedOcrModelError):
        resolve_bedrock_ocr_model("us.openai.gpt-6-astra", "ap-northeast-1")
