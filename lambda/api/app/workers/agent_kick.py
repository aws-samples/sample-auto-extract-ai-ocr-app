"""Agent Kick Lambda ハンドラー

Step Functions Map 内で ProcessImage の後に実行される。
agent_enabled を確認し、有効であれば AgentCore Runtime で検証を実行する。
"""
import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from repositories import get_image
from repositories.image_repository import get_images_table
from repositories.schema_repository import get_app_schema
from services.agent_service import AgentService

logger = logging.getLogger(__name__)


def _update_agent_status(image_id: str, status: str):
    """image レコードの agent_status を更新"""
    try:
        get_images_table().update_item(
            Key={"id": image_id},
            UpdateExpression="SET agent_status = :s",
            ExpressionAttributeValues={":s": status},
        )
    except Exception as e:
        logger.warning(f"Failed to update agent_status for {image_id}: {e}")


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
        _update_agent_status(image_id, "skipped")
        return {"status": "skipped", "reason": "no app_name"}

    # Check agent_enabled (avoid unnecessary job creation)
    schema = get_app_schema(app_name)
    if not schema or not schema.get("agent_enabled", False):
        logger.info(f"Agent not enabled for app: {app_name}")
        _update_agent_status(image_id, "skipped")
        return {"status": "skipped", "reason": "agent_enabled=false"}

    # Run agent correction directly (not via start_agent_correction to avoid re-invoke)
    try:
        from repositories.job_repository import create_agent_job, update_agent_job
        # Always create a new agent job (don't reuse OCR job_id from Step Functions)
        job_id = create_agent_job(image_id)
        _update_agent_status(image_id, "processing")
        agent_service = AgentService()
        asyncio.run(agent_service._process_agent_correction_async(job_id, image_id))
        _update_agent_status(image_id, "completed")
        return {"image_id": image_id, "status": "completed", "job_id": job_id}
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        _update_agent_status(image_id, "failed")
        # Update job record so polling doesn't hang
        try:
            update_agent_job(job_id, "failed", error=str(e))
        except Exception:
            pass
        return {"image_id": image_id, "status": "failed", "error": str(e)}
