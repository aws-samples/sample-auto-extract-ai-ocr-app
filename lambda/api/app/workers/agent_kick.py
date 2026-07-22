"""Agent Kick Lambda ハンドラー

Step Functions Map 内で ProcessImage の後に実行される。
agent_enabled を確認し、有効であれば AgentCore Runtime で検証を実行する。
"""
import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from repositories import get_image, update_agent_status
from repositories.job_repository import create_agent_job, update_agent_job, get_job
from repositories.schema_repository import get_app_schema
from services.agent_service import AgentService
from services.pdf_conversion_service import sync_parent_agent_status
from domains.image_status import AgentStatus
from repositories.job_repository import JobStatus

logger = logging.getLogger(__name__)


def _update_agent_status(image_id: str, status: str):
    """image レコードの agent_status を更新"""
    try:
        update_agent_status(image_id, status)
    except Exception as e:
        logger.warning(f"Failed to update agent_status for {image_id}: {e}")


def _finalize_skipped_job(event: dict):
    """If a pre-created job_id was passed, mark it as skipped so polling doesn't hang."""
    job_id = event.get("job_id")
    if not job_id:
        return
    try:
        update_agent_job(job_id, JobStatus.SKIPPED)
    except Exception as e:
        logger.warning(f"Failed to finalize skipped job {job_id}: {e}")


def agent_kick_handler(event, context):
    """Step Functions 用: 1 枚の画像に対して Agent 検証を実行

    Args:
        event: {"image_id": str, "job_id": str}

    Returns:
        {"image_id": str, "status": str, "job_id": str (optional)}
    """
    image_id = event.get("image_id")
    if not image_id:
        return {"status": "skipped", "reason": "no image_id"}

    # Get image record
    image_data = get_image(image_id)
    if not image_data:
        return {"status": "skipped", "reason": "image not found"}

    app_name = image_data.get("app_name", "")
    if not app_name:
        _update_agent_status(image_id, AgentStatus.SKIPPED)
        _finalize_skipped_job(event)
        sync_parent_agent_status(image_id)
        return {"status": "skipped", "reason": "no app_name"}

    # 手動実行（manual=True）は agent_enabled のみ要求し、自動実行判定を通さない。
    # 自動実行（Step Functions 経由）は agent_enabled かつ agent_auto_run のときのみ。
    is_manual = event.get("manual", False)
    schema = get_app_schema(app_name)
    enabled = bool(schema and schema.get("agent_enabled", False))
    allowed = enabled if is_manual else (enabled and bool(schema.get("agent_auto_run", False)))
    if not allowed:
        reason = "agent_enabled=false" if is_manual else "agent_auto_run=false"
        logger.info(f"Agent skipped for app {app_name} (manual={is_manual}): {reason}")
        # 検証対象外のユースケースは「検証していない」状態＝idle にする。
        # SKIPPED にすると GET /agent が過去ジョブを返し古い結果が復活するため。
        _update_agent_status(image_id, AgentStatus.IDLE)
        _finalize_skipped_job(event)
        sync_parent_agent_status(image_id)
        return {"status": "skipped", "reason": reason}

    job_id = event.get("job_id")
    try:
        if not job_id:
            job_id = create_agent_job(image_id)
        _update_agent_status(image_id, AgentStatus.PROCESSING)
        sync_parent_agent_status(image_id)
        agent_service = AgentService()
        asyncio.run(agent_service._process_agent_correction_async(job_id, image_id))
        # Re-read job to check actual status (may be failed internally)
        job = get_job(job_id)
        job_status = job.get("status", "failed") if job else "failed"
        if job_status == JobStatus.COMPLETED:
            suggestions_count = len(job.get("suggestions", [])) if job else 0
            update_agent_status(image_id, AgentStatus.COMPLETED, suggestions_count=suggestions_count)
            sync_parent_agent_status(image_id)
            return {"image_id": image_id, "status": "completed", "job_id": job_id}
        else:
            _update_agent_status(image_id, AgentStatus.FAILED)
            sync_parent_agent_status(image_id)
            return {"image_id": image_id, "status": "failed", "job_id": job_id}
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        _update_agent_status(image_id, AgentStatus.FAILED)
        sync_parent_agent_status(image_id)
        if job_id:
            try:
                update_agent_job(job_id, JobStatus.FAILED, error=str(e))
            except Exception:
                pass
        return {"image_id": image_id, "status": "failed", "error": str(e)}
