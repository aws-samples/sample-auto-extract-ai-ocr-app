from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
import logging
from typing import Optional

from services.s3_sync_service import S3SyncService
from dependencies.services import get_s3_sync_service
from dependencies.auth import get_cognito_sub, RequirePermission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/s3-sync", tags=["S3 Sync"])


class S3ImportRequest(BaseModel):
    bucket: str
    key: str
    filename: str
    page_processing_mode: str = "combined"


@router.post("/{app_name}")
async def sync_s3_files(app_name: str, prefix: Optional[str] = None, user=Depends(RequirePermission("viewer")), service: S3SyncService = Depends(get_s3_sync_service)):
    """S3バケットからファイルを同期する"""
    try:
        return await service.sync_s3_files(app_name, prefix)
    except Exception as e:
        logger.error(f"Error syncing S3 files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/{app_name}/import")
async def import_s3_file(app_name: str, body: S3ImportRequest, req: Request, user=Depends(RequirePermission("editor")), service: S3SyncService = Depends(get_s3_sync_service)):
    """S3バケットからファイルをインポートしてOCR処理を開始する"""
    try:
        sub = get_cognito_sub(req)
        return await service.import_s3_file(app_name, body.model_dump(), uploaded_by=sub)
    except Exception as e:
        logger.error(f"Error importing S3 file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/{app_name}/list")
async def list_s3_files_with_duplicate_check(app_name: str, prefix: Optional[str] = None, user=Depends(RequirePermission("viewer")), service: S3SyncService = Depends(get_s3_sync_service)):
    """S3ファイル一覧を重複チェック付きで取得する"""
    try:
        return await service.get_files_with_duplicate_check(app_name, prefix)
    except Exception as e:
        logger.error(f"Error listing S3 files with duplicate check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
