"""Schema Generate Worker Lambda ハンドラー

同期の POST /apps/{app_name}/schema/generate が API Gateway の 29 秒制限を
超えて 504 になっていた問題を解決するため、非同期化した Worker。
API Lambda が async invoke でこのハンドラーを起動する。

event: {"job_id": str, "s3_key": str, "filename": str, "instructions": str}
成功時:  JobsTable に status=completed + result={"fields": [...]}
失敗時:  JobsTable に status=failed + error=str(e)
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from repositories.job_repository import update_schema_generation_job
from schemas import SchemaGenerateRequest
from services.schema_service import SchemaService

logger = logging.getLogger(__name__)


def schema_generate_handler(event, context):
    """Worker: 1 件のスキーマ生成ジョブを実行する

    Args:
        event: {"job_id": str, "s3_key": str, "filename": str, "instructions": str}

    Returns:
        {"job_id": str, "status": "completed" | "failed"}
    """
    job_id = event.get("job_id")
    if not job_id:
        logger.error("schema_generate_handler: job_id is missing in event")
        return {"status": "failed", "reason": "no job_id"}

    s3_key = event.get("s3_key")
    filename = event.get("filename")
    instructions = event.get("instructions", "")

    if not s3_key or not filename:
        error_msg = "s3_key or filename is missing in event"
        logger.error(f"schema_generate_handler: {error_msg}")
        try:
            update_schema_generation_job(job_id, "failed", error=error_msg)
        except Exception as e:
            logger.warning(f"Failed to update job {job_id}: {e}")
        return {"job_id": job_id, "status": "failed"}

    try:
        request = SchemaGenerateRequest(
            s3_key=s3_key,
            filename=filename,
            instructions=instructions,
        )
        service = SchemaService()

        # 既存の generate_schema (async) を同期的に実行
        schema = asyncio.run(service.generate_schema(request))

        update_schema_generation_job(job_id, "completed", result=schema)
        logger.info(f"Schema generation job {job_id} completed")
        return {"job_id": job_id, "status": "completed"}
    except Exception as e:
        logger.error(f"Schema generation job {job_id} failed: {e}")
        try:
            update_schema_generation_job(job_id, "failed", error=str(e))
        except Exception as update_err:
            logger.error(f"Failed to update failed job {job_id}: {update_err}")
        return {"job_id": job_id, "status": "failed", "error": str(e)}
