from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# OCR関連の型定義
class OcrWord(BaseModel):
    """OCR認識された単語の情報"""
    id: int
    content: str
    rec_score: Optional[float] = None
    points: Optional[List[List[float]]] = None
    page: Optional[int] = None
    direction: Optional[int] = None


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


class OcrStartRequest(BaseModel):
    """OCR処理開始リクエスト"""
    app_name: Optional[str] = None


# ファイルアップロード関連の型定義
class PresignedUrlRequest(BaseModel):
    """プリサインドURL取得リクエスト"""
    filename: str
    content_type: str
    app_name: str = "default"
    page_processing_mode: str = "combined"


class PresignedUrlResponse(BaseModel):
    """プリサインドURL取得レスポンス"""
    presigned_url: str
    s3_key: str
    image_id: str


class UploadCompleteRequest(BaseModel):
    """アップロード完了通知リクエスト"""
    image_id: str
    filename: str
    s3_key: str
    app_name: str = "default"
    page_processing_mode: str = "combined"


# 情報抽出関連の型定義
class ExtractionRequest(BaseModel):
    """情報抽出リクエスト"""
    image_id: str
    app_name: Optional[str] = None
    words: Optional[List[OcrWord]] = None


class ExtractionResult(BaseModel):
    """情報抽出結果"""
    extracted_data: Dict[str, Any]
    status: str
    error: Optional[str] = None


# スキーマ関連の型定義
class SchemaField(BaseModel):
    """スキーマフィールド定義"""
    name: str
    type: str
    description: Optional[str] = None
    required: bool = False


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


# ジョブ関連の型定義
class JobStatus(BaseModel):
    """ジョブステータス"""
    job_id: str
    status: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class JobStartResponse(BaseModel):
    """ジョブ開始レスポンス"""
    jobId: str


# 画像関連の型定義
class ImageInfo(BaseModel):
    """画像情報"""
    id: str
    filename: str
    s3_key: str
    status: str
    upload_time: Optional[str] = None
    app_name: Optional[str] = None
    page_processing_mode: Optional[str] = None


class ImageListResponse(BaseModel):
    """画像リストレスポンス"""
    images: List[ImageInfo]
    total_count: int


# アプリ管理関連の型定義
class AppCreateRequest(BaseModel):
    """アプリ作成リクエスト"""
    app_name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    extraction_fields: Optional[List[Dict[str, Any]]] = None
    custom_prompt: Optional[str] = None


class AppUpdateRequest(BaseModel):
    """アプリ更新リクエスト"""
    display_name: Optional[str] = None
    description: Optional[str] = None
    extraction_fields: Optional[List[Dict[str, Any]]] = None
    custom_prompt: Optional[str] = None
    input_methods: Optional[Dict[str, bool]] = None


class CustomPromptRequest(BaseModel):
    """カスタムプロンプト更新リクエスト"""
    custom_prompt: str


# エラーレスポンス
class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    error: str
    detail: Optional[str] = None
    status_code: int


# 汎用レスポンス
class SuccessResponse(BaseModel):
    """成功レスポンス"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
