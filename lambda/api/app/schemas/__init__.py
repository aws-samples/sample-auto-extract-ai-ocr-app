"""
Pydantic schemas for API request/response validation
"""
from .ocr import OcrWord, OcrResult, OcrResultResponse, OcrStartRequest
from .upload import PresignedUrlRequest, PresignedUrlResponse, UploadCompleteRequest
from .schema import SchemaGenerateRequest, SchemaSaveRequest
from .job import JobStartResponse
from .image import ImageInfo
from .app import CustomPromptRequest

__all__ = [
    # OCR
    "OcrWord",
    "OcrResult",
    "OcrResultResponse",
    "OcrStartRequest",
    # Upload
    "PresignedUrlRequest",
    "PresignedUrlResponse",
    "UploadCompleteRequest",
    # Schema
    "SchemaGenerateRequest",
    "SchemaSaveRequest",
    # Job
    "JobStartResponse",
    # Image
    "ImageInfo",
    # App
    "CustomPromptRequest",
]
