"""AgentCore Runtime クライアント"""
import json
import logging
import uuid
from config import settings
from .aws import bedrock_agentcore_client

logger = logging.getLogger(__name__)


class AgentClient:
    """AgentCore Runtime 呼び出し専用クライアント"""

    def __init__(self):
        self.runtime_arn = settings.AGENT_RUNTIME_ARN
        self.client = bedrock_agentcore_client

    async def invoke_agent(
        self,
        messages: list,
        system_prompt: str,
        prompt: str,
        model_info: dict,
        allowed_tool_names: list[str] | None = None,
        image_content: list[dict] | None = None,
    ) -> str:
        """AgentCore Runtime を呼び出してレスポンステキストを返す"""
        try:
            input_data = {
                "messages": messages,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "model": model_info,
            }
            if allowed_tool_names:
                input_data["allowed_tool_names"] = allowed_tool_names
            if image_content:
                input_data["image_content"] = image_content

            payload = json.dumps({"input": input_data})
            session_id = "s-" + str(uuid.uuid4()).replace("-", "")
            response = self.client.invoke_agent_runtime(
                agentRuntimeArn=self.runtime_arn,
                runtimeSessionId=session_id,
                payload=payload,
            )
            response_body = response["response"].read()
            response_data = json.loads(response_body)
            return self._parse_response(response_data)
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            raise

    def _parse_response(self, response_data: dict) -> str:
        """AgentCore Runtime レスポンスからテキストを抽出

        Runtime returns: {"output": {"result": {"message": {...}, "status": "success"}, "timestamp": "..."}}
        message is Strands AgentResult.message: {"role": "assistant", "content": [{"text": "..."}]}
        """
        try:
            if isinstance(response_data, dict):
                output = response_data.get("output", {})
                result = output.get("result", {})

                # Check for error status
                if result.get("status") == "error":
                    return f"Agent error: {result.get('error', 'unknown')}"

                message = result.get("message", {})

                # message is a Bedrock Message dict: {"role": "...", "content": [...]}
                if isinstance(message, dict):
                    content = message.get("content", [])
                    if isinstance(content, list):
                        # Extract all text blocks
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                texts.append(block["text"])
                        if texts:
                            return "\n".join(texts)

                # Fallback: message might be a plain string
                if isinstance(message, str):
                    return message

                # Last resort
                return str(result) if result else str(response_data)
            return str(response_data)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return f"Error parsing response: {str(e)}"
