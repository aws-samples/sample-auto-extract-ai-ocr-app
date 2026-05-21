"""Main FastAPI application for OCR Agent Runtime."""

import json
import logging
import traceback
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.agent import AgentManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="OCR Agent Runtime",
    description="AWS Bedrock AgentCore Runtime with Strands Agent",
    version="2.0.0",
)

agent_manager = AgentManager()


@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ocr-agent-runtime"}


@app.post("/invocations")
async def invocations(request: Request):
    """Main invocation endpoint"""
    try:
        body = await request.body()
        body_str = body.decode()
        request_data = json.loads(body_str)

        logger.info("Received request")

        # Handle input field if present
        if "input" in request_data and isinstance(request_data["input"], dict):
            request_data = request_data["input"]

        # Extract fields
        prompt = request_data.get("prompt", "")
        messages = request_data.get("messages", [])
        system_prompt = request_data.get("system_prompt")
        model_info = request_data.get("model", {})
        allowed_tool_names = request_data.get("allowed_tool_names")
        image_content = request_data.get("image_content")

        # Process request
        result = agent_manager.process_request(
            messages=messages,
            system_prompt=system_prompt,
            prompt=prompt,
            model_info=model_info,
            allowed_tool_names=allowed_tool_names,
            image_content=image_content,
        )

        response = {
            "output": {
                "result": result,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
