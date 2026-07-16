from clients import s3_client
import boto3
import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Dict, Any

from schemas import (
    SchemaGenerateRequest, PresignedUrlRequest, CustomPromptRequest, PresignedUrlResponse, SchemaSaveRequest
)
from config import settings
from repositories import (
    get_app_schemas, get_app_schema, get_extraction_fields_for_app,
    get_custom_prompt_for_app, update_app_schema, create_app_schema,
    delete_app_schema, delete_images_by_app_name
)
from repositories.image_repository import create_s3_sync_folder
from repositories.usecase_repository import register_usecase_owner, delete_usecase_by_app_name, get_permitted_apps_with_permission
from repositories.job_repository import (
    JobType,
    create_schema_generation_job,
    get_job,
)
from domains.schema_generator import build_schema_generation_request, parse_schema_generation_response
from domains.schema_fields import extract_field_names
from clients.bedrock import call_bedrock
from utils.bedrock import parse_converse_response
from utils.pdf import pdf_page_to_jpeg

logger = logging.getLogger(__name__)


class SchemaService:
    """スキーマ・アプリ管理を行うサービスクラス"""

    def __init__(self):
        self.bucket_name = settings.BUCKET_NAME

    def _create_s3_sync_folder_if_needed(self, app_data: dict, app_name: str):
        """S3同期が有効な場合のみフォルダを作成"""
        if app_data.get("input_methods", {}).get("s3_sync", False):
            create_s3_sync_folder(app_name)

    def _build_app_data(self, request: SchemaSaveRequest) -> dict:
        """リクエストからapp_dataを構築"""
        return {
            "name": request.name,
            "display_name": request.display_name,
            "description": request.description or f"{request.display_name}からの情報抽出",
            # SchemaField(pydantic) → dict（既存の保存形と同じ。None キーは落とす）
            "fields": [f.model_dump(exclude_none=True) for f in request.fields],
            "input_methods": request.input_methods,
            "agent_enabled": request.agent_enabled,
            "sample_image_s3_key": request.sample_image_s3_key,
            "sample_image_filename": request.sample_image_filename,
            "schema_instructions": request.schema_instructions,
        }

    async def get_apps_list(self, user_id: str = None, role: str = None) -> Dict[str, Any]:
        """アプリ一覧を取得する（権限フィルタリング込み）

        Args:
            user_id: ユーザーID（None の場合はフィルタなし）
            role: ユーザーのシステムロール（admin の場合は全件 + owner 権限付与）
        """
        try:
            result = get_app_schemas()

            if role == "admin":
                for a in result.get("apps", []):
                    a["permission"] = "owner"
                return result

            if user_id is None:
                return result

            # 1クエリで許可済み app_name + permission を取得
            perm_map = get_permitted_apps_with_permission(user_id)
            apps = []
            for a in result.get("apps", []):
                perm = perm_map.get(a["name"])
                if perm:
                    a["permission"] = perm
                    apps.append(a)
            result["apps"] = apps
            return result
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
            field_names = extract_field_names(extraction_fields.get("fields", []))

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

    async def delete_app(self, app_name: str) -> None:
        """アプリを削除する"""
        try:
            # 1. DSQL のユースケース + 中間テーブルを削除（先に削除。失敗しても SchemasTable が残るので再削除可能）
            delete_usecase_by_app_name(app_name)

            # 2. 関連する画像データを削除（DynamoDB）
            delete_images_by_app_name(app_name)

            # 3. スキーマを削除（DynamoDB — マスタなので最後に削除）
            delete_app_schema(app_name)

            logger.info(f"Deleted app and related data: {app_name}")
        except Exception as e:
            logger.error(f"Error deleting app: {str(e)}")
            raise

    async def save_schema(self, request: SchemaSaveRequest, user_id: str) -> Dict[str, str]:
        """スキーマを保存する"""
        try:
            # 入力バリデーション
            if not request.name or not request.display_name:
                raise ValueError("アプリ名と表示名は必須です")

            # アプリ名のバリデーション（英数字とアンダースコアのみ）
            if not re.match(r'^[a-zA-Z0-9_]+$', request.name):
                raise ValueError("アプリ名は英数字とアンダースコアのみ使用できます")

            # 入力方法のバリデーション
            if not request.input_methods.get("file_upload", False) and not request.input_methods.get("s3_sync", False):
                raise ValueError("ファイルアップロードまたはS3同期のいずれかを有効にする必要があります")

            # スキーマデータを作成
            app_data = self._build_app_data(request)

            # S3同期が有効な場合、フォルダを作成
            self._create_s3_sync_folder_if_needed(app_data, request.name)

            # 1. DSQL に usecase + owner 権限を登録（先に実行。失敗すれば DynamoDB には書き込まない）
            register_usecase_owner(request.name, user_id)

            # 2. DynamoDB にスキーマ保存（DSQL 成功後。ConditionExpression で同名重複を原子的に防止）
            try:
                create_app_schema(request.name, app_data)
            except Exception:
                # DynamoDB 保存失敗時は DSQL 側を rollback して孤児を防ぐ
                logger.warning(f"DynamoDB save failed, rolling back DSQL usecase: {request.name}")
                try:
                    delete_usecase_by_app_name(request.name)
                except Exception:
                    logger.error(f"DSQL rollback also failed for: {request.name}")
                raise

            logger.info(f"Saved schema for app: {request.name}")
            return {"status": "success", "message": "スキーマが正常に保存されました"}
        except Exception as e:
            logger.error(f"Error saving schema: {str(e)}")
            raise

    async def get_sample_image_url(self, app_name: str) -> Dict[str, Any]:
        """スキーマに紐づくサンプル画像の presigned GET URL を返す

        サンプル画像が紐づいていない場合は url=None を返す (404 にはしない。
        紐付けは任意項目のため、フロントは url の有無で表示を切り替える)。
        """
        try:
            app_schema = get_app_schema(app_name)
            if not app_schema:
                raise ValueError(f"App '{app_name}' not found")

            s3_key = app_schema.get("sample_image_s3_key")
            if not s3_key:
                return {"url": None, "filename": None}

            # ContentType を取得 (PDF / 画像でフロントの表示を分けるため)
            try:
                head = s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                content_type = head.get("ContentType", "application/octet-stream")
            except Exception:
                # オブジェクトが削除済み等の場合は未紐付け扱い
                logger.warning(f"Sample image not found in S3: {s3_key}")
                return {"url": None, "filename": None}

            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key,
                    'ResponseContentType': content_type,
                },
                ExpiresIn=3600,  # 1時間
                HttpMethod='GET'
            )
            return {
                "url": presigned_url,
                "filename": app_schema.get("sample_image_filename"),
                "content_type": content_type,
                # 保存済み画像を使った再生成 (スキーマ生成 API へ渡す) 用
                "s3_key": s3_key,
            }
        except Exception as e:
            logger.error(f"Error getting sample image url: {str(e)}")
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
            _, ext = os.path.splitext(request.filename)
            ext = ext.lower()

            # PDFの場合は画像に変換
            if ext == '.pdf':
                try:
                    file_data = pdf_page_to_jpeg(file_data, page_num=0, dpi=300)
                    logger.info(f"PDFを画像に変換しました: {request.filename}")
                except Exception as e:
                    logger.error(f"PDF変換エラー: {str(e)}")
                    raise ValueError("PDFの変換に失敗しました。有効なPDFファイルをアップロードしてください。")
            elif ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                raise ValueError(
                    "サポートされていないファイル形式です。JPG、PNG、GIF、PDFのみ対応しています。")

            # スキーマフィールドを生成（build → Bedrock 呼び出し → parse）
            messages, system_prompts = build_schema_generation_request(
                file_data, request.instructions
            )
            response = call_bedrock(messages, system_prompts)
            fields_text = parse_converse_response(response)
            schema = parse_schema_generation_response(fields_text)

            # 常に {"fields": [...]} の形式で返す
            if "fields" not in schema:
                return {"fields": []}

            logger.info(f"Generated schema fields from {request.filename}")
            return schema
        except Exception as e:
            logger.error(f"Error generating schema: {str(e)}")
            raise

    async def start_schema_generation(self, request: SchemaGenerateRequest) -> Dict[str, str]:
        """スキーマ生成を非同期で開始する。

        - JobsTable にジョブを作成 (status=processing)
        - SchemaGenerate Worker Lambda を async invoke
        - job_id を即返却 (API Gateway の 29 秒制限回避のため)

        Args:
            request: スキーマ生成リクエスト (s3_key, filename, instructions)

        Returns:
            {"job_id": str, "status": "processing"}
        """
        try:
            if not settings.SCHEMA_GENERATE_FUNCTION_NAME:
                raise ValueError("SCHEMA_GENERATE_FUNCTION_NAME is not configured")

            # 1. ジョブレコード作成
            job_id = create_schema_generation_job(
                s3_key=request.s3_key,
                filename=request.filename,
                instructions=request.instructions or "",
            )
            logger.info(f"Created schema generation job: {job_id}")

            # 2. Worker Lambda を async invoke
            lambda_client = boto3.client("lambda")
            lambda_client.invoke(
                FunctionName=settings.SCHEMA_GENERATE_FUNCTION_NAME,
                InvocationType="Event",  # async
                Payload=json.dumps({
                    "job_id": job_id,
                    "s3_key": request.s3_key,
                    "filename": request.filename,
                    "instructions": request.instructions or "",
                }),
            )
            logger.info(f"Invoked SchemaGenerate Lambda for job {job_id}")

            return {"job_id": job_id, "status": "processing"}
        except Exception as e:
            logger.error(f"Error starting schema generation: {str(e)}")
            raise

    async def get_schema_generation_result(self, job_id: str) -> Dict[str, Any]:
        """スキーマ生成ジョブの状態と結果を取得する。

        Args:
            job_id: JobsTable の PK

        Returns:
            {"status": "processing" | "completed" | "failed",
             "result": {...} | None, "error": str | None}
        """
        job = get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.get("job_type") != JobType.SCHEMA_GENERATION:
            raise ValueError(f"Job {job_id} is not a schema_generation job")

        return {
            "status": job.get("status", "processing"),
            "result": job.get("result"),
            "error": job.get("error"),
        }

    async def update_schema(self, app_name: str, request: SchemaSaveRequest) -> Dict[str, str]:
        """既存のスキーマを更新する"""
        try:
            # 入力バリデーション
            if not request.name or not request.display_name:
                raise ValueError("アプリ名と表示名は必須です")

            # アプリ名のバリデーション（英数字とアンダースコアのみ）
            if not re.match(r'^[a-zA-Z0-9_]+$', request.name):
                raise ValueError("アプリ名は英数字とアンダースコアのみ使用できます")

            # 入力方法のバリデーション
            if not request.input_methods.get("file_upload", False) and not request.input_methods.get("s3_sync", False):
                raise ValueError("ファイルアップロードまたはS3同期のいずれかを有効にする必要があります")

            # スキーマデータを作成
            app_data = self._build_app_data(request)

            # S3同期が有効な場合、フォルダを作成
            self._create_s3_sync_folder_if_needed(app_data, app_name)

            # スキーマを更新
            update_app_schema(app_name, app_data)

            logger.info(f"Updated schema for app: {app_name}")
            return {"status": "success", "message": f"アプリ '{app_name}' を更新しました"}
        except Exception as e:
            logger.error(f"Error updating schema: {str(e)}")
            raise
