from clients import s3_client
import logging
import uuid
from datetime import datetime
from typing import Dict, Any

from schemas import (
    SchemaGenerateRequest, PresignedUrlRequest, CustomPromptRequest, PresignedUrlResponse, SchemaSaveRequest
)
from config import settings
from app_schema import (
    get_app_schemas, get_app_schema, get_extraction_fields_for_app,
    get_field_names_for_app, get_custom_prompt_for_app, update_app_schema,
    delete_app_schema, generate_schema_fields_from_image
)

logger = logging.getLogger(__name__)


class SchemaService:
    """スキーマ・アプリ管理を行うサービスクラス"""

    def __init__(self):
        self.bucket_name = settings.BUCKET_NAME

    async def get_apps_list(self) -> Dict[str, Any]:
        """アプリ一覧を取得する"""
        try:
            return get_app_schemas()
        except Exception as e:
            logger.error(f"Error getting apps list: {str(e)}")
            raise

    async def get_app_details(self, app_name: str) -> Dict[str, Any]:
        """アプリ詳細を取得する"""
        try:
            app_schemas = get_app_schemas()
            for app in app_schemas.get("apps", []):
                if app["name"] == app_name:
                    return app
            raise ValueError(f"App '{app_name}' not found")
        except Exception as e:
            logger.error(f"Error getting app details: {str(e)}")
            raise

    async def get_app_fields(self, app_name: str) -> Dict[str, Any]:
        """アプリのフィールド一覧を取得する"""
        try:
            extraction_fields = get_extraction_fields_for_app(app_name)
            field_names = get_field_names_for_app(app_name)

            return {
                "app_name": app_name,
                "extraction_fields": extraction_fields,
                "field_names": field_names
            }
        except Exception as e:
            logger.error(f"Error getting app fields: {str(e)}")
            raise

    async def get_custom_prompt(self, app_name: str) -> Dict[str, str]:
        """カスタムプロンプトを取得する"""
        try:
            custom_prompt = get_custom_prompt_for_app(app_name)
            return {"custom_prompt": custom_prompt}
        except Exception as e:
            logger.error(f"Error getting custom prompt: {str(e)}")
            raise

    async def update_custom_prompt(self, app_name: str, request: CustomPromptRequest) -> None:
        """カスタムプロンプトを更新する"""
        try:
            # 既存のアプリスキーマを取得
            app_schema = get_app_schema(app_name)
            if not app_schema:
                raise ValueError(f"App '{app_name}' not found")

            # カスタムプロンプトを更新
            app_schema["custom_prompt"] = request.custom_prompt

            # スキーマを保存
            update_app_schema(app_name, app_schema)

            logger.info(f"Updated custom prompt for app {app_name}")
        except Exception as e:
            logger.error(f"Error updating custom prompt: {str(e)}")
            raise

    async def create_app(self, app_data: dict) -> Dict[str, str]:
        """新しいアプリを作成または更新する"""
        try:
            app_name = app_data.get("name")
            if not app_name:
                raise ValueError("アプリ名が指定されていません")

            # 必須フィールドの検証
            required_fields = ["display_name", "fields"]
            for field in required_fields:
                if field not in app_data:
                    raise ValueError(f"必須フィールドがありません: {field}")

            # アプリスキーマを更新
            update_app_schema(app_name, app_data)

            logger.info(f"Created/updated app: {app_name}")
            return {"status": "success", "message": f"アプリ '{app_name}' を作成/更新しました"}
        except Exception as e:
            logger.error(f"Error creating app: {str(e)}")
            raise

    async def delete_app(self, app_name: str) -> None:
        """アプリを削除する"""
        try:
            delete_app_schema(app_name)
            logger.info(f"Deleted app: {app_name}")
        except Exception as e:
            logger.error(f"Error deleting app: {str(e)}")
            raise

    async def save_schema(self, request: SchemaSaveRequest) -> Dict[str, str]:
        """スキーマを保存する"""
        try:
            # 入力バリデーション
            if not request.name or not request.display_name:
                raise ValueError("アプリ名と表示名は必須です")

            # アプリ名のバリデーション（英数字とアンダースコアのみ）
            import re
            if not re.match(r'^[a-zA-Z0-9_]+$', request.name):
                raise ValueError("アプリ名は英数字とアンダースコアのみ使用できます")

            # 入力方法のバリデーション
            if not request.input_methods.get("file_upload", False) and not request.input_methods.get("s3_sync", False):
                raise ValueError("ファイルアップロードまたはS3同期のいずれかを有効にする必要があります")

            # S3同期が有効な場合、S3 URIが必要
            if request.input_methods.get("s3_sync", False) and not request.input_methods.get("s3_uri"):
                raise ValueError("S3同期が有効な場合、S3 URIを指定する必要があります")

            # スキーマデータを作成
            app_data = {
                "name": request.name,
                "display_name": request.display_name,
                "description": request.description or f"{request.display_name}からの情報抽出",
                "fields": request.fields,
                "input_methods": request.input_methods
            }

            # スキーマを保存
            update_app_schema(request.name, app_data)

            logger.info(f"Saved schema for app: {request.name}")
            return {"status": "success", "message": "スキーマが正常に保存されました"}
        except Exception as e:
            logger.error(f"Error saving schema: {str(e)}")
            raise

    async def generate_schema_presigned_url(self, request: PresignedUrlRequest) -> PresignedUrlResponse:
        """スキーマ用の署名付きURLを生成する"""
        try:
            # 一意のS3キーを生成
            image_id = str(uuid.uuid4())
            s3_key = f"schema-uploads/{datetime.now().isoformat()}_{request.filename}"

            # 署名付きURLの生成（有効期限は15分）
            presigned_url = s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key,
                    'ContentType': request.content_type
                },
                ExpiresIn=900,  # 15分
            )

            logger.info(
                f"Generated schema presigned URL for {request.filename}")

            return PresignedUrlResponse(
                presigned_url=presigned_url,
                image_id=image_id,
                s3_key=s3_key
            )
        except Exception as e:
            logger.error(f"Error generating schema presigned URL: {str(e)}")
            raise

    async def generate_schema(self, request: SchemaGenerateRequest) -> Dict[str, Any]:
        """スキーマを自動生成する"""
        try:
            # S3からファイルを取得
            try:
                s3_response = s3_client.get_object(
                    Bucket=settings.BUCKET_NAME,
                    Key=request.s3_key
                )
                file_data = s3_response['Body'].read()
            except Exception as e:
                logger.error(f"S3からのファイル取得エラー: {str(e)}")
                raise ValueError("ファイルが見つかりません")

            # ファイルの種類を拡張子で判定
            import os
            _, ext = os.path.splitext(request.filename)
            ext = ext.lower()

            # PDFの場合は画像に変換
            if ext == '.pdf':
                try:
                    import fitz
                    pdf_document = fitz.open(stream=file_data, filetype="pdf")
                    if pdf_document.page_count > 0:
                        page = pdf_document[0]
                        # 高解像度で変換
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(300/72, 300/72))
                        file_data = pix.tobytes("jpeg")
                        logger.info(f"PDFを画像に変換しました: {request.filename}")
                    else:
                        raise ValueError("PDFにページがありません")
                    pdf_document.close()
                except Exception as e:
                    logger.error(f"PDF変換エラー: {str(e)}")
                    raise ValueError("PDFの変換に失敗しました。有効なPDFファイルをアップロードしてください。")
            elif ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                raise ValueError(
                    "サポートされていないファイル形式です。JPG、PNG、GIF、PDFのみ対応しています。")

            # スキーマフィールドを生成
            schema = generate_schema_fields_from_image(
                file_data,
                request.instructions
            )

            # 常に {"fields": [...]} の形式で返す
            if "fields" not in schema:
                return {"fields": []}

            logger.info(f"Generated schema fields from {request.filename}")
            return schema
        except Exception as e:
            logger.error(f"Error generating schema: {str(e)}")
            raise

    async def update_schema(self, app_name: str, request: SchemaSaveRequest) -> Dict[str, str]:
        """既存のスキーマを更新する"""
        try:
            # 入力バリデーション
            if not request.name or not request.display_name:
                raise ValueError("アプリ名と表示名は必須です")

            # アプリ名のバリデーション（英数字とアンダースコアのみ）
            import re
            if not re.match(r'^[a-zA-Z0-9_]+$', request.name):
                raise ValueError("アプリ名は英数字とアンダースコアのみ使用できます")

            # 入力方法のバリデーション
            if not request.input_methods.get("file_upload", False) and not request.input_methods.get("s3_sync", False):
                raise ValueError("ファイルアップロードまたはS3同期のいずれかを有効にする必要があります")

            # S3同期が有効な場合、S3 URIが必要
            if request.input_methods.get("s3_sync", False) and not request.input_methods.get("s3_uri"):
                raise ValueError("S3同期が有効な場合、S3 URIを指定する必要があります")

            # スキーマデータを作成
            app_data = {
                "name": request.name,
                "display_name": request.display_name,
                "description": request.description or f"{request.display_name}からの情報抽出",
                "fields": request.fields,
                "input_methods": request.input_methods
            }

            # スキーマを更新
            update_app_schema(app_name, app_data)

            logger.info(f"Updated schema for app: {app_name}")
            return {"status": "success", "message": f"アプリ '{app_name}' を更新しました"}
        except Exception as e:
            logger.error(f"Error updating schema: {str(e)}")
            raise
