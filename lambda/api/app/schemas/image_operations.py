"""Schemas for image operation requests (process, verification, agent)"""
from typing import List, Literal
from pydantic import BaseModel


class ProcessRequest(BaseModel):
    # OCR をスキップして抽出から実行する（既存 OCR 結果を再利用）
    skip_ocr: bool = False


class VerificationRequest(BaseModel):
    verification_completed: bool = False


class SuggestionStatusUpdate(BaseModel):
    status: Literal["accepted", "rejected"]


class S3ImportItem(BaseModel):
    bucket: str
    key: str
    filename: str


class S3ImportBatchRequest(BaseModel):
    files: List[S3ImportItem]
    page_processing_mode: Literal["combined", "individual"] = "combined"
