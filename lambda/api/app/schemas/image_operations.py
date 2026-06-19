"""Schemas for image operation requests (process, verification, agent)"""
from typing import Optional, Literal
from pydantic import BaseModel


class ProcessRequest(BaseModel):
    start_from: Optional[str] = None


class VerificationRequest(BaseModel):
    verification_completed: bool = False


class SuggestionStatusUpdate(BaseModel):
    status: Literal["accepted", "rejected"]


class S3ImportRequest(BaseModel):
    bucket: str
    key: str
    filename: str
    page_processing_mode: str = "combined"
