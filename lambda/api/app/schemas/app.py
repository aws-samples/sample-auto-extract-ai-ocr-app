from pydantic import BaseModel


class CustomPromptRequest(BaseModel):
    """カスタムプロンプト更新リクエスト"""
    custom_prompt: str
