"""Agent management for the agent runtime."""

import logging
from typing import Any

import boto3
from strands import Agent as StrandsAgent
from strands.models import BedrockModel

from .config import extract_model_info, get_max_iterations, get_system_prompt
from .tools import create_gateway_mcp_client
from .types import Message, ModelInfo
from .utils import process_messages, process_prompt

logger = logging.getLogger(__name__)


class IterationLimitExceededError(Exception):
    """Exception raised when iteration limit is exceeded"""
    pass


class AgentManager:
    """Manages Strands agent creation and execution"""

    def __init__(self):
        self.max_iterations = get_max_iterations()

    def _create_iteration_limit_handler(self):
        """Create a per-request iteration limit handler (thread-safe)"""
        state = {"count": 0}

        def handler(**ev):
            if ev.get("init_event_loop"):
                state["count"] = 0
            if ev.get("start_event_loop"):
                state["count"] += 1
                if state["count"] > self.max_iterations:
                    raise IterationLimitExceededError(
                        f"Event loop reached maximum iteration count ({self.max_iterations})"
                    )
        return handler

    def _build_prompt_with_images(
        self,
        prompt: str | list[dict[str, Any]],
        image_content: list[dict[str, Any]] | None,
    ) -> str | list[dict[str, Any]]:
        """Build prompt with image content blocks if provided."""
        if not image_content:
            return prompt

        # Build content blocks: images first, then text prompt
        import base64
        content_blocks = []
        for img in image_content:
            image_bytes = base64.b64decode(img["bytes_base64"])
            content_blocks.append({
                "image": {
                    "format": img.get("format", "jpeg"),
                    "source": {"bytes": image_bytes},
                }
            })

        # Add text prompt
        if isinstance(prompt, str):
            content_blocks.append({"text": prompt})
        elif isinstance(prompt, list):
            content_blocks.extend(prompt)

        return content_blocks

    def process_request(
        self,
        messages: list[Message] | list[dict[str, Any]],
        system_prompt: str | None,
        prompt: str | list[dict[str, Any]],
        model_info: ModelInfo,
        allowed_tool_names: list[str] | None = None,
        image_content: list[dict[str, Any]] | None = None,
    ) -> dict:
        """Process a request and return complete response"""
        try:
            model_id, region = extract_model_info(model_info)
            combined_system_prompt = get_system_prompt(system_prompt)

            # Gateway MCP client with optional tool filtering
            mcp_client = create_gateway_mcp_client(allowed_tool_names)
            tools = [mcp_client] if mcp_client else []

            session = boto3.Session(region_name=region)
            bedrock_model = BedrockModel(
                model_id=model_id,
                boto_session=session,
            )

            processed_messages = process_messages(messages)
            processed_prompt = process_prompt(prompt)
            final_prompt = self._build_prompt_with_images(processed_prompt, image_content)

            agent = StrandsAgent(
                system_prompt=combined_system_prompt,
                messages=processed_messages,
                model=bedrock_model,
                tools=tools,
                callback_handler=self._create_iteration_limit_handler(),
            )

            result = agent(final_prompt)

            return {
                "message": result.message if hasattr(result, 'message') else str(result),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error processing agent request: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
