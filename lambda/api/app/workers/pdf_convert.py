"""PDF Convert Worker Lambda ハンドラー

アップロード完了後の PDF→画像変換を担う Worker。API Lambda が async invoke で起動する。
API Lambda 内のバックグラウンドスレッドで変換していた旧方式は、HTTP 応答後に
実行環境が回収されると変換が失われ画像が CONVERTING のまま残る取りこぼしがあった。
独立 Lambda に切り出すことで、infra 障害時は async invoke のリトライが効く。

event: {"image_id": str, "s3_key": str}
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.pdf_conversion_service import convert_pdf_to_image

logger = logging.getLogger(__name__)


def pdf_convert_handler(event, context):
    """Worker: 1 件の PDF 変換を実行する

    Args:
        event: {"image_id": str, "s3_key": str}

    Returns:
        {"image_id": str, "status": "done" | "failed"}
    """
    image_id = event.get("image_id")
    s3_key = event.get("s3_key")

    if not image_id or not s3_key:
        logger.error(f"pdf_convert_handler: image_id or s3_key is missing in event: {event}")
        return {"image_id": image_id, "status": "failed"}

    # convert_pdf_to_image は自前で例外を握り画像を FAILED に更新するため re-raise 不要。
    convert_pdf_to_image(image_id, s3_key)
    return {"image_id": image_id, "status": "done"}
