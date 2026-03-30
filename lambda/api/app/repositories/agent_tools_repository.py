"""Agent ツール Repository（DynamoDB ToolsTable）

AgentCore Runtime が登録するツール一覧を管理する。
DSQL の tools テーブル（tool_repository.py）とは別物。
"""
import logging
import os
from clients import dynamodb_resource

logger = logging.getLogger(__name__)

_tools_table = None


def _get_table():
    global _tools_table
    if _tools_table is None:
        table_name = os.environ.get("TOOLS_TABLE_NAME", "")
        if not table_name:
            logger.warning("TOOLS_TABLE_NAME not set")
            return None
        _tools_table = dynamodb_resource.Table(table_name)
    return _tools_table


def list_agent_tools() -> list[dict]:
    """AgentCore Runtime のツール一覧を取得"""
    table = _get_table()
    if not table:
        return []
    try:
        response = table.scan()
        items = response.get("Items", [])
        return [
            {"name": item.get("tool_name", ""), "description": item.get("description", "")}
            for item in items
        ]
    except Exception as e:
        logger.error(f"Error getting agent tools from DynamoDB: {e}")
        return []
