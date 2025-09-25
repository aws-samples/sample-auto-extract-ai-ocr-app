from fastapi import APIRouter, HTTPException
import logging
from typing import Optional

from services.s3_sync_service import S3SyncService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/s3-sync", tags=["S3 Sync"])

# S3同期サービスのインスタンス
s3_sync_service = S3SyncService()


@router.post("/{app_name}")
async def sync_s3_files(app_name: str, prefix: Optional[str] = None):
    """S3バケットからファイルを同期する"""
    try:
        result = await s3_sync_service.sync_s3_files(app_name, prefix)
        return result
    except Exception as e:
        logger.error(f"Error syncing S3 files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/{app_name}/import")
async def import_s3_file(app_name: str, file_data: dict):
    """S3バケットからファイルをインポートしてOCR処理を開始する"""
    try:
        result = await s3_sync_service.import_s3_file(app_name, file_data)
        return result
    except Exception as e:
        logger.error(f"Error importing S3 file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
