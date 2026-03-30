from pydantic import BaseModel, Field
from typing import Optional, List


class ImageInfo(BaseModel):
    """画像情報（API レスポンス用 — camelCase alias 付き）

    DynamoDB の snake_case キーを受け取り、serialization_alias で camelCase に変換する。
    """
    id: str
    name: Optional[str] = Field(None, alias="filename", serialization_alias="name")
    s3_key: Optional[str | list] = None
    upload_time: Optional[str] = Field(None, serialization_alias="uploadTime")
    status: Optional[str] = None
    job_id: Optional[str] = Field(None, serialization_alias="jobId")
    app_name: Optional[str] = Field(None, serialization_alias="appName")
    page_processing_mode: Optional[str] = Field(None, serialization_alias="pageProcessingMode")
    total_pages: Optional[int] = Field(None, serialization_alias="totalPages")
    page_number: Optional[int] = Field(None, serialization_alias="pageNumber")
    parent_document_id: Optional[str] = Field(None, serialization_alias="parentDocumentId")
    verification_completed: Optional[bool] = Field(False, serialization_alias="verificationCompleted")
    uploaded_by: Optional[str] = None
    uploaded_by_email: Optional[str] = None
    verified_by: Optional[str] = None
    verified_by_email: Optional[str] = None

    model_config = {"populate_by_name": True}


class ImageListResponse(BaseModel):
    """画像リストレスポンス"""
    images: List[ImageInfo]
    total: int
