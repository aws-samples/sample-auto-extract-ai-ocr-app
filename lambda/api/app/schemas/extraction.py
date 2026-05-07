from pydantic import BaseModel
from typing import Optional, List
from .ocr import OcrWord


class ExtractionRequest(BaseModel):
    """情報抽出リクエスト"""
    image_id: str
    app_name: Optional[str] = None
    words: Optional[List[OcrWord]] = None
