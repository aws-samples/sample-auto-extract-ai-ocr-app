"""textract OCR エンジンのディスパッチと呼び出しのテスト。

想定している正しい挙動:
- OCR_ENGINE=textract のとき _invoke_ocr は SageMaker を使わず _invoke_textract_ocr に
  ディスパッチし、SAGEMAKER_ENDPOINT_NAME が無くても動く。
- DetectDocumentText に画像バイトを渡し、LINE ブロックを元画像のピクセル座標の words にする。
- 送信前に縮小しても、座標は元画像の寸法で計算する。圧縮の上限は Textract の 10MB。
- JPEG / PNG はそのまま、それ以外（TIFF 等）は JPEG に変換して送る。
- Textract 例外時は他エンジンと同じエラー dict 形状を返す。
- get_endpoint_status は常に ready、_check_endpoint_ready は SageMaker を確認しない。
"""
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image

import services.ocr_service as ocr_module
from services.ocr_service import OcrService


def _image_bytes(fmt="JPEG", size=(200, 100)):
    buf = BytesIO()
    Image.new("RGB", size).save(buf, format=fmt)
    return buf.getvalue()


class _FakeTextract:
    def __init__(self, response=None, error=None):
        self.response = response or {"Blocks": []}
        self.error = error
        self.calls = []

    def detect_document_text(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _make_service(monkeypatch, fake=None):
    monkeypatch.setattr(ocr_module.settings, "ENABLE_OCR", True)
    monkeypatch.setattr(ocr_module.settings, "OCR_ENGINE", "textract")
    monkeypatch.setattr(ocr_module.settings, "SAGEMAKER_ENDPOINT_NAME", "")
    fake = fake or _FakeTextract()
    monkeypatch.setattr(ocr_module, "textract_client", fake)
    return OcrService(), fake


def _line(text, left, top, width, height):
    return {
        "BlockType": "LINE",
        "Text": text,
        "Geometry": {"BoundingBox": {"Left": left, "Top": top, "Width": width, "Height": height}},
    }


class TestInvokeOcrDispatch:
    def test_textract_engine_dispatches_without_sagemaker_endpoint(self, monkeypatch):
        svc, _ = _make_service(monkeypatch)
        called = {}
        monkeypatch.setattr(
            svc, "_invoke_textract_ocr",
            lambda image_data: called.setdefault("data", image_data) or {"text": "", "words": [], "word_count": 0},
        )

        svc._invoke_ocr(b"img")

        assert called["data"] == b"img"


class TestInvokeTextractOcr:
    def test_success_returns_ocr_contract(self, monkeypatch):
        fake = _FakeTextract({"Blocks": [_line("請求書", 0.1, 0.1, 0.5, 0.2)]})
        svc, _ = _make_service(monkeypatch, fake)
        image = _image_bytes()

        result = svc._invoke_textract_ocr(image)

        assert fake.calls == [{"Document": {"Bytes": image}}]
        assert result == {
            "text": "請求書",
            "words": [{"id": 0, "content": "請求書",
                       "points": [[20, 10], [120, 10], [120, 30], [20, 30]]}],
            "word_count": 1,
        }

    def test_points_use_original_size_after_downscale(self, monkeypatch):
        fake = _FakeTextract({"Blocks": [_line("合計", 0.5, 0.5, 0.25, 0.25)]})
        svc, _ = _make_service(monkeypatch, fake)
        received = {}

        def fake_compress(image_data, max_bytes=None):
            received["max_bytes"] = max_bytes
            return _image_bytes(size=(100, 50))

        monkeypatch.setattr(ocr_module, "compress_image_for_payload", fake_compress)

        result = svc._invoke_textract_ocr(_image_bytes(size=(200, 100)))

        assert received["max_bytes"] == ocr_module.TEXTRACT_IMAGE_MAX_BYTES == 10 * 1024 * 1024
        assert result["words"][0]["points"] == [[100, 50], [150, 50], [150, 75], [100, 75]]

    def test_png_is_sent_as_is(self, monkeypatch):
        svc, fake = _make_service(monkeypatch)

        svc._invoke_textract_ocr(_image_bytes(fmt="PNG"))

        assert fake.calls[0]["Document"]["Bytes"].startswith(b"\x89PNG")

    def test_unsupported_format_is_converted_to_jpeg(self, monkeypatch):
        svc, fake = _make_service(monkeypatch)

        svc._invoke_textract_ocr(_image_bytes(fmt="TIFF"))

        assert fake.calls[0]["Document"]["Bytes"].startswith(b"\xff\xd8")

    def test_textract_exception_returns_error_dict(self, monkeypatch):
        svc, _ = _make_service(monkeypatch, _FakeTextract(error=Exception("throttled")))

        result = svc._invoke_textract_ocr(_image_bytes())

        assert result == {"error": result["error"], "text": "", "words": [], "word_count": 0}
        assert "throttled" in result["error"]


class TestEndpointStatusBypass:
    def test_endpoint_status_ready_for_textract(self, monkeypatch):
        svc, _ = _make_service(monkeypatch)
        assert svc.get_endpoint_status() == {"ready": True, "status": "textract"}

    def test_check_endpoint_ready_skips_sagemaker(self, monkeypatch):
        svc, _ = _make_service(monkeypatch)

        def fail():
            raise AssertionError("should not be called")

        monkeypatch.setattr(svc, "get_endpoint_status", fail)
        svc._check_endpoint_ready(skip_ocr=False)
