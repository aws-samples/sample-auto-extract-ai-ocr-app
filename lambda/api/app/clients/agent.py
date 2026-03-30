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

    async def invoke_agent(self, messages: list, system_prompt: str, prompt: str, model_info: dict) -> str:
        """AgentCore Runtime を呼び出してレスポンステキストを返す"""
        try:
            payload = json.dumps({
                "input": {
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "model": model_info,
                }
            })
            session_id = str(uuid.uuid4()).replace("-", "") + str(uuid.uuid4()).replace("-", "")[:1]
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
        """AgentCore Runtime レスポンスからテキストを抽出"""
        try:
            if isinstance(response_data, dict):
                output = response_data.get("output", {})
                result = output.get("result", {})
                message = result.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", [])
                    if isinstance(content, list) and len(content) > 0:
                        first_content = content[0]
                        if isinstance(first_content, dict):
                            return first_content.get("text", "")
                if "output" in response_data:
                    return str(output)
                elif "result" in response_data:
                    return str(result)
            return str(response_data)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return f"Error parsing response: {str(e)}"
