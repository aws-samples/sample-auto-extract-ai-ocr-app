from pydantic import BaseModel


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
    filename: str
    s3_key: str
    app_name: str = "default"
    page_processing_mode: str = "combined"
