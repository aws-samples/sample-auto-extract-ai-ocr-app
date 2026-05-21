"""Gateway ↔ DSQL Tool Sync (EventBridge triggered)

Triggered by CloudTrail events (CreateGatewayTarget, UpdateGatewayTarget,
DeleteGatewayTarget) to synchronize Gateway tools with DSQL tools table.
- New tools in Gateway → INSERT into DSQL
- Removed tools from Gateway → Physical DELETE from DSQL (cascade)
- Existing tools → UPDATE name/description
"""

import json
import logging
import os
import time

import boto3
import httpx
import psycopg2
import psycopg2.extras
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DSQL_ENDPOINT = os.environ.get("DSQL_ENDPOINT", "")
DSQL_REGION = os.environ.get("DSQL_REGION", "")
GATEWAY_ENDPOINT = os.environ.get("GATEWAY_ENDPOINT", "")
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")


def get_dsql_connection():
    """Get DSQL connection with IAM auth"""
    client = boto3.client("dsql", region_name=DSQL_REGION)
    token = client.generate_db_connect_admin_auth_token(DSQL_ENDPOINT, DSQL_REGION)
    return psycopg2.connect(
        host=DSQL_ENDPOINT,
        port=5432,
        user="admin",
        password=token,
        dbname="postgres",
        sslmode="require",
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def sign_request(method: str, url: str, headers: dict, body: bytes) -> dict:
    """Sign request with SigV4 for bedrock-agentcore service"""
    session = boto3.Session()
    credentials = session.get_credentials().get_frozen_credentials()
    aws_request = AWSRequest(
        method=method,
        url=url,
        headers=headers,
        data=body,
    )
    SigV4Auth(credentials, "bedrock-agentcore", AWS_REGION).add_auth(aws_request)
    return dict(aws_request.headers)


def list_gateway_tools() -> list[dict]:
    """Call Gateway's tools/list via MCP JSON-RPC with retry"""
    if not GATEWAY_ENDPOINT:
        raise RuntimeError("GATEWAY_ENDPOINT not configured")

    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/list",
    }
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            signed_headers = sign_request("POST", GATEWAY_ENDPOINT, headers, body)
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    GATEWAY_ENDPOINT,
                    content=body,
                    headers=signed_headers,
                )
                response.raise_for_status()

                data = response.json()
                tools = data.get("result", {}).get("tools", [])
                logger.info(f"Got {len(tools)} tools from Gateway")
                return tools

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError("Failed to list Gateway tools after retries")


def sync_tools_to_dsql(gateway_tools: list[dict]):
    """Sync Gateway tools to DSQL tools table.

    - New tools → INSERT
    - Deleted tools → Physical DELETE (with cascade to user_tools, group_tools, usecase_tools)
    - Existing tools → UPDATE description
    """
    conn = get_dsql_connection()
    try:
        conn.autocommit = False

        with conn.cursor() as cur:
            # Get existing tools from DSQL
            cur.execute("SELECT id, name, description FROM tools")
            existing_tools = {row["name"]: row for row in cur.fetchall()}

        # Guard: if Gateway returned 0 tools but DSQL has existing tools, skip sync
        if not gateway_tools and existing_tools:
            logger.warning(
                "Gateway returned 0 tools but DSQL has %d tools. "
                "Skipping sync to prevent accidental deletion.",
                len(existing_tools),
            )
            return

        # Build set of Gateway tool names
        gateway_tool_map = {}
        for tool in gateway_tools:
            tool_name = tool.get("name", "")
            if tool_name:
                gateway_tool_map[tool_name] = {
                    "name": tool_name,
                    "description": tool.get("description", ""),
                }

        with conn.cursor() as cur:
            # INSERT new tools
            for tool_name, tool_data in gateway_tool_map.items():
                if tool_name not in existing_tools:
                    cur.execute(
                        "INSERT INTO tools (name, description) VALUES (%s, %s)",
                        (tool_data["name"], tool_data["description"]),
                    )
                    logger.info(f"Inserted tool: {tool_name}")

            # UPDATE existing tools (description only)
            for tool_name, tool_data in gateway_tool_map.items():
                if tool_name in existing_tools:
                    existing = existing_tools[tool_name]
                    if existing["description"] != tool_data["description"]:
                        cur.execute(
                            "UPDATE tools SET description = %s WHERE name = %s",
                            (tool_data["description"], tool_name),
                        )
                        logger.info(f"Updated tool: {tool_name}")

            # DELETE removed tools (cascade)
            for tool_name, existing in existing_tools.items():
                if tool_name not in gateway_tool_map:
                    tool_id = str(existing["id"])
                    cur.execute("DELETE FROM user_tools WHERE tool_id = %s", (tool_id,))
                    cur.execute("DELETE FROM group_tools WHERE tool_id = %s", (tool_id,))
                    cur.execute("DELETE FROM usecase_tools WHERE tool_id = %s", (tool_id,))
                    cur.execute("DELETE FROM tools WHERE id = %s", (tool_id,))
                    logger.info(f"Deleted tool: {tool_name} (id={tool_id})")

        conn.commit()
        logger.info("Tool sync completed successfully")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error syncing tools: {e}")
        raise
    finally:
        conn.close()


def handler(event, context):
    """Lambda handler for both CustomResource and EventBridge triggers.

    - CustomResource: triggered on CDK deploy (initial sync + target additions via CDK)
    - EventBridge: triggered by CloudTrail events (CreateGatewayTarget, UpdateGatewayTarget, DeleteGatewayTarget)
    """
    logger.info(f"Event: {json.dumps(event)}")

    # CustomResource invocation
    if "RequestType" in event:
        if event["RequestType"] == "Delete":
            return {"Status": "SUCCESS", "PhysicalResourceId": "tool-sync"}

        try:
            gateway_tools = list_gateway_tools()
            sync_tools_to_dsql(gateway_tools)
            return {
                "Status": "SUCCESS",
                "PhysicalResourceId": "tool-sync",
                "Data": {"ToolCount": len(gateway_tools)},
            }
        except Exception as e:
            logger.error(f"CustomResource failed: {e}")
            return {
                "Status": "FAILED",
                "PhysicalResourceId": "tool-sync",
                "Reason": str(e),
            }

    # EventBridge invocation
    event_name = event.get("detail", {}).get("eventName", "")
    logger.info(f"Triggered by: {event_name}")

    gateway_tools = list_gateway_tools()
    sync_tools_to_dsql(gateway_tools)

    return {"status": "success", "toolCount": len(gateway_tools)}
