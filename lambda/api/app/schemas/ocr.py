from pydantic import BaseModel
from typing import Optional, List, Dict


class OcrWord(BaseModel):
    """OCR認識された単語の情報"""
    id: int
    content: str
    rec_score: Optional[float] = None
    points: Optional[List[List[float]]] = None
    page: Optional[int] = None
    direction: Optional[str] = None


class OcrResult(BaseModel):
    """OCR処理結果"""
    words: List[OcrWord]
    text: Optional[str] = None
    word_count: Optional[int] = None
    total_pages: Optional[int] = None
    pages: Optional[List[Dict]] = None
    error: Optional[str] = None


class OcrResultResponse(BaseModel):
    """OCR結果取得APIのレスポンス"""
    filename: Optional[str]
    s3_key: Optional[str]
    uploadTime: Optional[str]
    status: Optional[str]
    ocrResult: OcrResult
    imageUrl: Optional[str]
    app_name: Optional[str]


class OcrStartRequest(BaseModel):
    """OCR処理開始リクエスト"""
    app_name: Optional[str] = None
    image_ids: Optional[List[str]] = None
    skip_ocr: bool = False
