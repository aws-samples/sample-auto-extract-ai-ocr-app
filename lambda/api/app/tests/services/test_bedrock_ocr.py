"""bedrock（LLM）OCR エンジンのディスパッチと呼び出しのテスト。

想定している正しい挙動:
- OCR_ENGINE=bedrock のとき _invoke_ocr は SageMaker を使わず _invoke_bedrock_ocr に
  ディスパッチし、SAGEMAKER_ENDPOINT_NAME が無くても動く。
- _invoke_bedrock_ocr は Converse の画像上限（3.5MB）で圧縮し、Converse 非対応
  フォーマット（TIFF 等）は JPEG に変換して送る。
- 応答の JSON パース失敗時、応答が JSON の断片（"{" や "```" 始まり）でなければ
  改行分割（座標なし）で回復し、JSON 断片や空応答はエラー dict にする。
- Bedrock 例外時は SageMaker パスと同じエラー dict 形状を返す。
- get_endpoint_status / _check_endpoint_ready は bedrock のとき SageMaker を確認しない。
"""
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from PIL import Image

import services.ocr_service as ocr_module
from services.ocr_service import OcrService
from exceptions import ResponseParseError


def _image_bytes(fmt="JPEG", size=(200, 100)):
    buf = BytesIO()
    Image.new("RGB", size).save(buf, format=fmt)
    return buf.getvalue()


def _make_bedrock_service(monkeypatch):
    monkeypatch.setattr(ocr_module.settings, "ENABLE_OCR", True)
    monkeypatch.setattr(ocr_module.settings, "OCR_ENGINE", "bedrock")
    monkeypatch.setattr(ocr_module.settings, "SAGEMAKER_ENDPOINT_NAME", "")
    return OcrService()


def _install_call(monkeypatch, response_text):
    """call_bedrock_and_parse を、固定の応答テキストで parse_fn を呼ぶフェイクに差し替える"""
    captured = {}

    def fake_call(messages, system_prompts, parse_fn, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return parse_fn(response_text)

    monkeypatch.setattr(ocr_module, "call_bedrock_and_parse", fake_call)
    return captured


class TestInvokeOcrDispatch:
    def test_bedrock_engine_dispatches_without_sagemaker_endpoint(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)
        called = {}
        monkeypatch.setattr(
            svc, "_invoke_bedrock_ocr",
            lambda image_data: called.setdefault("data", image_data) or {"text": "", "words": [], "word_count": 0},
        )

        svc._invoke_ocr(b"img")

        assert called["data"] == b"img"

    def test_paddle_engine_without_endpoint_raises(self, monkeypatch):
        monkeypatch.setattr(ocr_module.settings, "ENABLE_OCR", True)
        monkeypatch.setattr(ocr_module.settings, "OCR_ENGINE", "paddle")
        monkeypatch.setattr(ocr_module.settings, "SAGEMAKER_ENDPOINT_NAME", "")
        svc = OcrService()

        with pytest.raises(RuntimeError):
            svc._invoke_ocr(b"img")


class TestInvokeBedrockOcr:
    def test_success_returns_ocr_contract(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)
        captured = _install_call(
            monkeypatch, '{"words": [{"text": "請求書", "box": [10, 10, 50, 20]}]}')

        result = svc._invoke_bedrock_ocr(_image_bytes())

        assert result["word_count"] == 1
        assert result["words"][0]["content"] == "請求書"
        assert result["words"][0]["points"] == [[10, 10], [50, 10], [50, 20], [10, 20]]
        # 画像とプロンプト（寸法入り）が送られる
        content = captured["messages"][0]["content"]
        assert content[0]["image"]["format"] == "jpeg"
        assert "幅200px・高さ100px" in content[1]["text"]

    def test_compresses_to_converse_image_limit(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)
        _install_call(monkeypatch, '{"words": []}')
        received = {}

        def fake_compress(image_data, max_bytes=None):
            received["max_bytes"] = max_bytes
            return image_data

        monkeypatch.setattr(ocr_module, "compress_image_for_payload", fake_compress)

        svc._invoke_bedrock_ocr(_image_bytes())

        assert received["max_bytes"] == ocr_module.BEDROCK_IMAGE_MAX_BYTES
        assert ocr_module.BEDROCK_IMAGE_MAX_BYTES < int(3.75 * 1024 * 1024)

    def test_unsupported_format_is_converted_to_jpeg(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)
        captured = _install_call(monkeypatch, '{"words": []}')

        svc._invoke_bedrock_ocr(_image_bytes(fmt="TIFF"))

        image_block = captured["messages"][0]["content"][0]["image"]
        assert image_block["format"] == "jpeg"
        # 送信バイトも JPEG に変換されている
        assert image_block["source"]["bytes"].startswith(b"\xff\xd8")

    def test_png_format_detected(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)
        captured = _install_call(monkeypatch, '{"words": []}')

        svc._invoke_bedrock_ocr(_image_bytes(fmt="PNG"))

        assert captured["messages"][0]["content"][0]["image"]["format"] == "png"

    def test_parse_error_falls_back_to_newline_split(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)

        def fake_call(messages, system_prompts, parse_fn, **kwargs):
            # 実物と同様、応答テキストで parse_fn を呼んだ結果 ResponseParseError になる
            return parse_fn("行1のテキスト\n行2のテキスト\n")

        monkeypatch.setattr(ocr_module, "call_bedrock_and_parse", fake_call)

        result = svc._invoke_bedrock_ocr(_image_bytes())

        assert result["word_count"] == 2
        assert result["words"][0] == {"id": 0, "content": "行1のテキスト"}
        assert "error" not in result

    def test_broken_json_is_not_recovered_as_text(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)

        def fake_call(messages, system_prompts, parse_fn, **kwargs):
            # maxTokens 打ち切り等で JSON が途中で切れたケース
            return parse_fn('{"words": [{"text": "abc", "box": [10,')

        monkeypatch.setattr(ocr_module, "call_bedrock_and_parse", fake_call)

        result = svc._invoke_bedrock_ocr(_image_bytes())

        # JSON 断片を OCR 結果として保存せず、エラーにする
        assert "error" in result
        assert result["words"] == []

    def test_parse_error_without_text_returns_error_dict(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)

        def fake_call(messages, system_prompts, parse_fn, **kwargs):
            # parse_fn が一度も呼ばれずに失敗した場合（応答テキストなし）
            raise ResponseParseError("no response")

        monkeypatch.setattr(ocr_module, "call_bedrock_and_parse", fake_call)

        result = svc._invoke_bedrock_ocr(_image_bytes())

        assert "error" in result
        assert result["words"] == []

    def test_bedrock_exception_returns_error_dict(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)

        def fake_call(*args, **kwargs):
            raise Exception("throttled")

        monkeypatch.setattr(ocr_module, "call_bedrock_and_parse", fake_call)

        result = svc._invoke_bedrock_ocr(_image_bytes())

        assert "error" in result
        assert result == {
            "error": result["error"], "text": "", "words": [], "word_count": 0,
        }


class TestEndpointStatusBypass:
    def test_endpoint_status_ready_for_bedrock(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)
        assert svc.get_endpoint_status() == {"ready": True, "status": "llm_ocr"}

    def test_check_endpoint_ready_skips_sagemaker(self, monkeypatch):
        svc = _make_bedrock_service(monkeypatch)

        def fail():
            raise AssertionError("should not be called")

        monkeypatch.setattr(svc, "get_endpoint_status", fail)
        # 例外なく通ること
        svc._check_endpoint_ready(skip_ocr=False)
