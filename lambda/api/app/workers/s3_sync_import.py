"""S3 Sync Import Worker Lambda ハンドラー

S3 同期インポートの重い処理（コピー → リサイズ / PDF 変換キックオフ）を担う Worker。
画像レコードは API 側で作成・重複チェック済み。

event: {"app_name": str, "page_processing_mode": str,
        "items": [{"image_id", "source_bucket", "source_key", "destination_key", "filename"}, ...]}
"""
import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from clients import s3_client
from config import settings
from domains.image_status import ImageStatus
from repositories.image_repository import update_image_status
from schemas import UploadCompleteRequest
from services.upload_service import UploadService

logger = logging.getLogger(__name__)

# OCR パイプラインの Step Functions Map(maxConcurrency 5)と並列度を揃える。
MAX_CONCURRENCY = 5


def _process_item(item: dict, app_name: str, page_processing_mode: str) -> bool:
    """1 ファイルを処理する。成功で True。例外は握って画像を FAILED にする。"""
    image_id = item.get("image_id")
    try:
        s3_client.copy_object(
            CopySource={"Bucket": item["source_bucket"], "Key": item["source_key"]},
            Bucket=settings.BUCKET_NAME,
            Key=item["destination_key"],
        )

        # 直接アップロードと同じフローを再利用（リサイズ / PDF 変換キックオフ）
        request = UploadCompleteRequest(
            filename=item["filename"],
            s3_key=item["destination_key"],
            app_name=app_name,
            page_processing_mode=page_processing_mode,
        )
        asyncio.run(UploadService().handle_upload_complete(image_id, request))
        return True
    except Exception as e:
        logger.error(f"S3 import failed for image {image_id}: {e}")
        try:
            update_image_status(image_id, ImageStatus.FAILED)
        except Exception as update_err:
            logger.error(f"Failed to mark image {image_id} FAILED: {update_err}")
        return False


def s3_sync_import_handler(event, context):
    """Worker: S3 インポートのバッチを並列処理する

    Returns:
        {"app_name": str, "total": int, "succeeded": int, "failed": int}
    """
    app_name = event.get("app_name")
    page_processing_mode = event.get("page_processing_mode", "combined")
    items = event.get("items", [])

    if not app_name or not items:
        logger.error(f"s3_sync_import_handler: app_name or items missing in event: {event}")
        return {"app_name": app_name, "total": 0, "succeeded": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        results = list(executor.map(
            lambda item: _process_item(item, app_name, page_processing_mode),
            items,
        ))

    succeeded = sum(1 for r in results if r)
    failed = len(results) - succeeded
    logger.info(f"S3 import batch for {app_name}: {succeeded} succeeded, {failed} failed")
    return {"app_name": app_name, "total": len(results), "succeeded": succeeded, "failed": failed}
