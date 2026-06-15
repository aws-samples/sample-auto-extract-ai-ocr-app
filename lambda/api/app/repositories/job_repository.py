from clients import dynamodb_resource
import logging
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from datetime import datetime
import uuid
from config import settings

logger = logging.getLogger(__name__)


def get_jobs_table():
    """ジョブテーブルのリソースを取得する"""
    table_name = settings.JOBS_TABLE_NAME
    if not table_name:
        logger.error("JOBS_TABLE_NAME 環境変数が設定されていません")
        raise ValueError("JOBS_TABLE_NAME environment variable is not set")
    return dynamodb_resource.Table(table_name)


def get_images_table():
    """画像テーブルのリソースを取得する（job_repository内で使用）"""
    table_name = settings.IMAGES_TABLE_NAME
    if not table_name:
        logger.error("IMAGES_TABLE_NAME 環境変数が設定されていません")
        raise ValueError("IMAGES_TABLE_NAME environment variable is not set")
    return dynamodb_resource.Table(table_name)


def get_job(job_id):
    """ジョブ情報を取得する。見つからない場合は None を返す。"""
    table = get_jobs_table()
    try:
        response = table.get_item(Key={"id": job_id})
        return response.get("Item")
    except ClientError as e:
        logger.error(f"ジョブ取得エラー: {str(e)}")
        raise


def get_latest_agent_job_by_image_id(image_id: str) -> dict | None:
    """image_id から最新のエージェントジョブを取得"""
    from boto3.dynamodb.conditions import Attr
    table = get_jobs_table()
    try:
        response = table.query(
            IndexName="ImageIdIndex",
            KeyConditionExpression=Key("image_id").eq(image_id),
            FilterExpression=Attr("job_type").eq("agent_correction"),
            ScanIndexForward=False,
            Limit=10,
        )
        items = response.get("Items", [])
        return items[0] if items else None
    except ClientError as e:
        logger.error(f"image_id によるジョブ取得エラー: {str(e)}")
        raise


def create_agent_job(image_id: str):
    """Create agent correction job

    Args:
        image_id: Image ID

    Returns:
        str: Job ID
    """
    job_id = str(uuid.uuid4())
    table = get_jobs_table()
    current_time = datetime.now().isoformat()

    try:
        item = {
            "id": job_id,
            "image_id": image_id,
            "job_type": "agent_correction",
            "status": "processing",
            "created_at": current_time,
            "updated_at": current_time
        }
        table.put_item(Item=item)
        return job_id
    except Exception as e:
        logger.error(f"Error creating agent job: {str(e)}")
        raise


def update_suggestion_status(image_id: str, suggestion_index: int, status: str) -> int:
    """Update a specific suggestion's status (accepted/rejected) and return pending count.

    Args:
        image_id: Image ID to find the latest agent job
        suggestion_index: Index of suggestion in the suggestions array
        status: New status ('accepted' or 'rejected')

    Returns:
        int: Number of remaining pending suggestions
    """
    job = get_latest_agent_job_by_image_id(image_id)
    if not job:
        raise ValueError("No agent job found for this image")

    suggestions = job.get("suggestions", [])
    if suggestion_index < 0 or suggestion_index >= len(suggestions):
        raise ValueError(f"Invalid suggestion index: {suggestion_index}")

    # Atomically update single element using DynamoDB path expression
    table = get_jobs_table()
    current_time = datetime.now().isoformat()
    table.update_item(
        Key={"id": job["id"]},
        UpdateExpression=f"SET suggestions[{suggestion_index}].#st = :status, updated_at = :u",
        ExpressionAttributeNames={"#st": "status"},
        ExpressionAttributeValues={
            ":status": status,
            ":u": current_time,
        },
    )

    # Count pending suggestions (after our update applied locally)
    suggestions[suggestion_index]["status"] = status
    pending_count = sum(1 for s in suggestions if s.get("status", "pending") == "pending")

    # Update image record with new pending count
    images_table = get_images_table()
    images_table.update_item(
        Key={"id": image_id},
        UpdateExpression="SET agent_suggestions_count = :c",
        ExpressionAttributeValues={":c": pending_count},
    )

    return pending_count


def update_agent_job(job_id: str, status: str, suggestions: list = None, error: str = None):
    """Update agent correction job

    Args:
        job_id: Job ID
        status: Job status (processing, completed, failed)
        suggestions: Correction suggestions
        error: Error message if failed
    """
    table = get_jobs_table()
    current_time = datetime.now().isoformat()

    try:
        update_expr = "SET #status = :status, updated_at = :updated_at"
        expr_attr_names = {"#status": "status"}
        expr_attr_values = {
            ":status": status,
            ":updated_at": current_time
        }

        if status == "completed":
            update_expr += ", completed_at = :completed_at"
            expr_attr_values[":completed_at"] = current_time

            if suggestions is not None:
                update_expr += ", suggestions = :suggestions"
                expr_attr_values[":suggestions"] = suggestions

        if error:
            update_expr += ", #error = :error"
            expr_attr_names["#error"] = "error"
            expr_attr_values[":error"] = error

        table.update_item(
            Key={"id": job_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_attr_names,
            ExpressionAttributeValues=expr_attr_values
        )
    except Exception as e:
        logger.error(f"Error updating agent job: {str(e)}")
        raise
