from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal


FieldType = Literal["string", "number", "map", "list"]


class FieldItems(BaseModel):
    """list 型フィールドの要素定義"""
    type: FieldType
    fields: Optional[List["SchemaField"]] = None  # 要素が map の場合の子フィールド
    model_config = {"extra": "forbid"}


class SchemaField(BaseModel):
    """抽出スキーマのフィールド定義（再帰構造）"""
    name: str
    display_name: str
    type: FieldType
    fields: Optional[List["SchemaField"]] = None  # map 型の子フィールド
    items: Optional[FieldItems] = None            # list 型の要素定義
    model_config = {"extra": "forbid"}


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
    fields: List[SchemaField]
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
