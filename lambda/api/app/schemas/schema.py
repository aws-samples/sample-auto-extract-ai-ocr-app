from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class SchemaGenerateRequest(BaseModel):
    """スキーマ生成リクエスト"""
    s3_key: str
    filename: str
    instructions: Optional[str] = None


class SchemaSaveRequest(BaseModel):
    """スキーマ保存リクエスト"""
    name: str
    display_name: str
    description: Optional[str] = None
    fields: List[Dict[str, Any]]
    input_methods: Dict[str, Any]
    agent_enabled: bool = False


class SchemaGenerateStartResponse(BaseModel):
    """スキーマ生成の非同期ジョブ起動レスポンス"""
    job_id: str
    status: str  # "processing"


class SchemaGenerateStatusResponse(BaseModel):
    """スキーマ生成ジョブの状態確認レスポンス"""
    status: str  # "processing" | "completed" | "failed"
    result: Optional[Dict[str, Any]] = None  # 完了時のみ {"fields": [...]}
    error: Optional[str] = None  # 失敗時のみ
