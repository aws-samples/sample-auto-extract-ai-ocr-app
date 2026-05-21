"""AgentCore Gateway Lambda Target: customer-lookup"""

import json
import logging
import os

import boto3
from boto3.dynamodb.conditions import Attr

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION"))
customers_table_name = os.environ.get("CUSTOMERS_TABLE", "")
customers_table = dynamodb.Table(customers_table_name) if customers_table_name else None


def search_customer_by_id(customer_id: str) -> dict:
    """顧客IDで顧客情報を検索"""
    if not customers_table:
        return {"error": "CUSTOMERS_TABLE not configured"}

    response = customers_table.get_item(Key={"customer_id": customer_id})
    item = response.get("Item", {})
    if item:
        logger.info(f"Found customer: {customer_id}")
        return item
    else:
        logger.info(f"Customer not found: {customer_id}")
        return {}


def search_customer_by_name(customer_name: str) -> list[dict]:
    """顧客名で顧客情報を検索（部分一致）"""
    if not customers_table:
        return [{"error": "CUSTOMERS_TABLE not configured"}]

    response = customers_table.scan(
        FilterExpression=Attr("customer_name").contains(customer_name)
    )
    items = response.get("Items", [])
    logger.info(f"Found {len(items)} customers matching '{customer_name}'")
    return items


def _get_tool_name(context) -> str:
    """Extract tool name from Lambda context (set by AgentCore Gateway)."""
    try:
        client_context = context.client_context
        if client_context and hasattr(client_context, "custom"):
            custom = client_context.custom or {}
            tool_name = custom.get("bedrockAgentCoreToolName", "")
            # Strip gateway target prefix (e.g., "customer-lookup___search_customer_by_name")
            if "___" in tool_name:
                return tool_name.split("___", 1)[1]
            return tool_name
    except Exception as e:
        logger.warning(f"Failed to extract tool name from context: {e}")
    return ""


def handler(event, context):
    """Lambda handler for AgentCore Gateway Target.

    Gateway passes tool arguments as the event payload.
    Tool name is provided via context.client_context.custom.bedrockAgentCoreToolName.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    tool_name = _get_tool_name(context)
    arguments = event
    logger.info(f"Tool name: {tool_name}")

    try:
        if tool_name == "search_customer_by_id":
            result = search_customer_by_id(arguments["customer_id"])
        elif tool_name == "search_customer_by_name":
            result = search_customer_by_name(arguments["customer_name"])
        else:
            return {"error": f"Unknown tool: {tool_name}"}

        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}
