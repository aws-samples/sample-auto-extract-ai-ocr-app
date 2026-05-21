"""Service for agent-based OCR correction."""

import asyncio
import base64
import json
import logging
from typing import Dict, Any, Optional

import boto3

from repositories import get_image
from repositories.job_repository import create_agent_job, update_agent_job, get_job
from repositories.tool_repository import list_tools, get_usecase_allowed_tool_names
from repositories.usecase_repository import get_usecase_by_app_name
from clients import AgentClient, s3_client
from config import settings
logger = logging.getLogger(__name__)


class AgentService:
    """Service for agent-based OCR correction suggestions"""

    def __init__(self):
        self.agent_client = AgentClient()
    
    async def start_agent_correction(self, image_id: str) -> str:
        """Start agent correction job
        
        Args:
            image_id: Image ID
            
        Returns:
            Job ID
        """
        try:
            # Create job
            job_id = create_agent_job(image_id)
            logger.info(f"Created agent job: {job_id} for image: {image_id}")
            
            # Invoke AgentKick Lambda asynchronously
            lambda_client = boto3.client("lambda")
            lambda_client.invoke(
                FunctionName=settings.AGENT_KICK_FUNCTION_NAME,
                InvocationType="Event",  # async
                Payload=json.dumps({"image_id": image_id, "job_id": job_id}),
            )
            logger.info(f"Invoked AgentKick Lambda for job {job_id}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting agent correction: {e}")
            raise
    
    def _process_agent_correction(self, job_id: str, image_id: str):
        """Process agent correction in background (sync wrapper)
        
        Args:
            job_id: Job ID
            image_id: Image ID
        """
        asyncio.run(self._process_agent_correction_async(job_id, image_id))
    
    async def _process_agent_correction_async(self, job_id: str, image_id: str):
        """Process agent correction in background

        Args:
            job_id: Job ID
            image_id: Image ID
        """
        try:
            logger.info(f"Processing agent correction for job: {job_id}, image: {image_id}")

            # Get OCR extraction results
            image_data = get_image(image_id)
            if not image_data:
                raise ValueError(f"Image not found: {image_id}")

            extracted_info = image_data.get("extracted_info", {})
            if not extracted_info:
                logger.warning(f"No extracted info found for image: {image_id}")
                update_agent_job(job_id, "completed", suggestions=[])
                return

            # Resolve allowed tool names from usecase (app_name → usecase_id → tools)
            app_name = image_data.get("app_name", "")
            usecase = get_usecase_by_app_name(app_name) if app_name else None
            usecase_id = str(usecase["id"]) if usecase else ""
            allowed_tool_names = get_usecase_allowed_tool_names(usecase_id) if usecase_id else []

            # Skip agent execution if no tools are configured
            if not allowed_tool_names:
                logger.info(f"No tools configured for usecase '{app_name}'. Skipping agent.")
                update_agent_job(job_id, "skipped")
                return

            # Fetch images from S3
            image_content = self._fetch_images(image_data)

            # Create system prompt
            system_prompt = self._create_system_prompt(extracted_info)

            # Call AgentCore Runtime
            response_text = await self.agent_client.invoke_agent(
                messages=[],
                system_prompt=system_prompt,
                prompt="OCR抽出結果を検証し、誤りがあれば修正してください。",
                model_info={
                    "modelId": settings.MODEL_ID,
                    "region": settings.MODEL_REGION
                },
                allowed_tool_names=allowed_tool_names or None,
                image_content=image_content or None,
            )

            # Parse response
            suggestions = self._parse_agent_response(response_text)

            # Update job as completed
            update_agent_job(job_id, "completed", suggestions=suggestions)
            logger.info(f"Agent correction completed for job: {job_id}")

        except Exception as e:
            logger.error(f"Error in agent correction job {job_id}: {e}")
            update_agent_job(job_id, "failed", error=str(e))
    
    async def get_agent_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get agent correction job status
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status and results
        """
        job = get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "suggestions": job.get("suggestions", []),
            "error": job.get("error"),
            "created_at": job.get("created_at"),
            "completed_at": job.get("completed_at")
        }
    
    async def get_available_tools(self) -> Dict[str, Any]:
        """Get available tools from DSQL

        Returns:
            Dictionary with tools list
        """
        try:
            tools = list_tools()
            return {"status": "success", "tools": tools}

        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            raise
    
    def _fetch_images(self, image_data: dict) -> list[dict]:
        """Fetch converted images from S3 and encode as base64"""
        converted_s3_keys = image_data.get("converted_s3_key", [])
        if isinstance(converted_s3_keys, str):
            converted_s3_keys = [converted_s3_keys]

        image_content = []
        for key in converted_s3_keys:
            try:
                response = s3_client.get_object(
                    Bucket=settings.BUCKET_NAME, Key=key
                )
                image_bytes = response["Body"].read()
                fmt = "jpeg"
                if key.lower().endswith(".png"):
                    fmt = "png"
                elif key.lower().endswith(".webp"):
                    fmt = "webp"
                image_content.append({
                    "format": fmt,
                    "bytes_base64": base64.b64encode(image_bytes).decode("utf-8"),
                })
            except Exception as e:
                logger.error(f"Failed to fetch image {key}: {e}")

        return image_content

    def _create_system_prompt(self, extracted_info: dict) -> str:
        """Create system prompt for agent
        
        Args:
            extracted_info: OCR extracted information
            
        Returns:
            System prompt string
        """
        return f"""あなたはOCR抽出結果を検証し、誤りを修正するアシスタントです。

## タスク
以下のOCR抽出結果を検証し、誤りがあれば修正してください。

## OCR抽出結果
{json.dumps(extracted_info, ensure_ascii=False, indent=2)}

## 指示
- 利用可能なツールを使って、抽出結果の正確性を検証してください
- 数値計算がある場合は、検算ツールを使用して計算の正確性を確認してください
- データベースに登録されている情報と照合し、不一致があれば修正案を提示してください

## 出力形式
修正が必要な場合のみ、以下のJSON形式で出力してください：
{{
  "suggestions": [
    {{
      "field": "フィールド名（例: client_info.address）",
      "original_value": "元の値",
      "suggested_value": "修正後の値",
      "reason": "修正理由",
      "confidence": "high" | "medium" | "low",
      "tool_used": "使用したツール名（例: get_customer_by_name）"
    }}
  ]
}}

修正が不要な場合は空の配列を返してください：
{{
  "suggestions": []
}}
"""
    

    def _parse_agent_response(self, response_text: str) -> list[dict]:
        """Parse agent response to extract suggestions
        
        Args:
            response_text: Agent response text
            
        Returns:
            List of suggestions
        """
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                else:
                    json_str = response_text
            
            result = json.loads(json_str)
            suggestions = result.get("suggestions", [])
            
            logger.info(f"Parsed {len(suggestions)} suggestions from agent response")
            return suggestions
        
        except Exception as e:
            logger.error(f"Error parsing agent response: {e}")
            logger.error(f"Response text: {response_text}")
            return []
    
