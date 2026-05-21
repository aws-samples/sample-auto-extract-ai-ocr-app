"""Gateway MCP Client for AgentCore Gateway Tools.

Creates MCP client with SigV4 authentication for Gateway tools.
Gateway uses AWS_IAM inbound auth — the Runtime's IAM Role signs requests.
"""

import logging
import os

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp import MCPClient

logger = logging.getLogger(__name__)


class SigV4HttpxAuth(httpx.Auth):
    """Runtime の IAM Role で Gateway リクエストに SigV4 署名"""

    def __init__(self, region: str | None = None, service: str = "bedrock-agentcore"):
        self.region = region or os.environ.get("AWS_REGION", "ap-northeast-1")
        self.service = service

    def auth_flow(self, request: httpx.Request):
        # 毎回最新の credentials を取得（コンテナ長時間稼働で期限切れ防止）
        session = boto3.Session()
        credentials = session.get_credentials().get_frozen_credentials()
        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            data=request.content,
        )
        SigV4Auth(credentials, self.service, self.region).add_auth(aws_request)
        request.headers.update(dict(aws_request.headers))
        yield request


class FilteredMCPClient(MCPClient):
    """MCPClient wrapper that filters tools based on allowed tool names.

    Gateway returns tools in "targetName___toolName" format.
    Filtering supports both full names and simplified names (after ___).
    """

    def __init__(self, client_factory, allowed_tool_names: list[str]):
        super().__init__(client_factory)
        self._allowed_tool_names = set(allowed_tool_names)
        logger.info(f"FilteredMCPClient created with {len(allowed_tool_names)} allowed tools")

    def list_tools_sync(self, *args, **kwargs):
        """List tools from Gateway and filter based on allowed_tool_names."""
        from strands.types import PaginatedList

        paginated_result = super().list_tools_sync(*args, **kwargs)

        filtered_tools = []
        for tool in paginated_result:
            full_name = tool.tool_name  # e.g., "customer-lookup___search_customer_by_id"

            # Extract simplified name (after ___) for matching
            if "___" in full_name:
                simplified_name = full_name.split("___", 1)[1]
            else:
                simplified_name = full_name

            # Allow if either full or simplified name is in the allowed set
            if full_name in self._allowed_tool_names or simplified_name in self._allowed_tool_names:
                filtered_tools.append(tool)

        logger.info(
            f"Filtered {len(filtered_tools)} tools from {len(paginated_result)} available"
        )
        return PaginatedList(filtered_tools, token=paginated_result.pagination_token)


def create_gateway_mcp_client(
    allowed_tool_names: list[str] | None = None,
) -> MCPClient | None:
    """SigV4 署名付き MCPClient を生成。

    Args:
        allowed_tool_names: 許可するツール名のリスト。None の場合は全ツールを公開。

    Returns:
        MCPClient instance or None if Gateway URL is not configured.
    """
    gateway_url = os.environ.get("AGENTCORE_GATEWAY_ENDPOINT")
    if not gateway_url:
        logger.warning("AGENTCORE_GATEWAY_ENDPOINT not set. Gateway tools disabled.")
        return None

    auth = SigV4HttpxAuth()

    if allowed_tool_names:
        client = FilteredMCPClient(
            lambda: streamablehttp_client(gateway_url, auth=auth),
            allowed_tool_names=allowed_tool_names,
        )
    else:
        client = MCPClient(lambda: streamablehttp_client(gateway_url, auth=auth))

    logger.info(f"Gateway MCP client created: {gateway_url}")
    return client
