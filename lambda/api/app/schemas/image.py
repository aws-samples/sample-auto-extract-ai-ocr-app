from pydantic import BaseModel, Field, field_validator
from typing import Optional

from domains.image_status import to_api_status


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
    agent_status: Optional[str] = Field(None, serialization_alias="agentStatus")
    agent_suggestions_count: Optional[int] = Field(None, serialization_alias="agentSuggestionsCount")
    uploaded_by: Optional[str] = None
    uploaded_by_email: Optional[str] = None
    verified_by: Optional[str] = None
    verified_by_email: Optional[str] = None

    model_config = {"populate_by_name": True}

    @field_validator("status")
    @classmethod
    def _fold_status(cls, v: Optional[str]) -> Optional[str]:
        # 内部の ocr/extracting はフロント向けに processing へ畳む
        return to_api_status(v)

