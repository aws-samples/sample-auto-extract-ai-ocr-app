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
    # スキーマ生成に使ったサンプル画像の S3 キー (schema-uploads/ 配下)。
    # None の場合、update 時は既存値を保持する (画像未変更の編集で消えないように)。
    sample_image_s3_key: Optional[str] = None
    sample_image_filename: Optional[str] = None
    # スキーマ生成に使った指示プロンプト。編集画面で復元し「プロンプトだけ変えて再生成」を可能にする。
    schema_instructions: Optional[str] = None


class SchemaGenerateStartResponse(BaseModel):
    """スキーマ生成の非同期ジョブ起動レスポンス"""
    job_id: str
    status: str  # "processing"


class SchemaGenerateStatusResponse(BaseModel):
    """スキーマ生成ジョブの状態確認レスポンス"""
    status: str  # "processing" | "completed" | "failed"
    result: Optional[Dict[str, Any]] = None  # 完了時のみ {"fields": [...]}
    error: Optional[str] = None  # 失敗時のみ
