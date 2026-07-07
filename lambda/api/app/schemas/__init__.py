"""
Pydantic schemas for API request/response validation
"""
from .ocr import OcrWord, OcrResult, OcrResultResponse, OcrStartRequest
from .upload import PresignedUrlRequest, PresignedUrlResponse, UploadCompleteRequest
from .schema import (
    SchemaGenerateRequest,
    SchemaSaveRequest,
    SchemaGenerateStartResponse,
    SchemaGenerateStatusResponse,
)
from .job import JobStartResponse
from .image import ImageInfo
from .app import CustomPromptRequest
from .image_operations import ProcessRequest, VerificationRequest, SuggestionStatusUpdate, S3ImportRequest
from .usecase import UsecaseToolsUpdate

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
    "SchemaGenerateStartResponse",
    "SchemaGenerateStatusResponse",
    # Job
    "JobStartResponse",
    # Image
    "ImageInfo",
    # App
    "CustomPromptRequest",
    # Image Operations
    "ProcessRequest",
    "VerificationRequest",
    "SuggestionStatusUpdate",
    "S3ImportRequest",
    # Usecase
    "UsecaseToolsUpdate",
]
