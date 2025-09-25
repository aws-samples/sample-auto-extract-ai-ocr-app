from clients import s3_client
import uuid
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi.responses import StreamingResponse
import io

from database import (
    create_image_record, get_image, get_images, update_image_status, update_converted_image
)
from schemas import (
    PresignedUrlRequest, PresignedUrlResponse, UploadCompleteRequest
)
from config import settings
from utils import resize_image, convert_pdf_to_image
from app_schema import get_app_schemas, get_app_input_methods

logger = logging.getLogger(__name__)

# 共通のS3クライアントを使用

DEFAULT_APP = "default"


class UploadService:
    """アップロード処理を管理するサービスクラス"""

    def __init__(self):
        self.bucket_name = settings.BUCKET_NAME

    async def generate_presigned_url(self, request: PresignedUrlRequest) -> PresignedUrlResponse:
        """署名付きURLを生成する"""
        try:
            # app_nameのバリデーション
            valid_app = False
            app_schemas = get_app_schemas()
            for app in app_schemas.get("apps", []):
                if app["name"] == request.app_name:
                    valid_app = True
                    break

            if not valid_app:
                logger.warning(
                    f"Invalid app name: {request.app_name}, using default: {DEFAULT_APP}")
                request.app_name = DEFAULT_APP

            # アプリケーションの入力方法設定を取得
            input_methods = get_app_input_methods(request.app_name)

            # ファイルアップロードが有効かチェック
            if not input_methods.get("file_upload", True):
                raise ValueError(
                    f"ファイルアップロードはこのアプリケーションでは無効です: {request.app_name}")

            # 一意のS3キーを生成
            image_id = str(uuid.uuid4())
            s3_key = f"uploads/{image_id}_{datetime.now().isoformat()}_{request.filename}"

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

            # DynamoDBにレコードを作成
            create_image_record(
                image_id=image_id,
                filename=request.filename,
                s3_key=s3_key,
                app_name=request.app_name,
                status="uploading",  # アップロード中ステータスを設定
                page_processing_mode=request.page_processing_mode  # 追加
            )

            logger.info(
                f"Generated presigned URL for {request.filename} (ID: {image_id})")

            return PresignedUrlResponse(
                presigned_url=presigned_url,
                image_id=image_id,
                s3_key=s3_key
            )

        except Exception as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            raise

    async def handle_upload_complete(self, request: UploadCompleteRequest) -> Dict[str, Any]:
        """アップロード完了を処理する"""
        try:
            # S3オブジェクトの存在確認
            try:
                s3_response = s3_client.head_object(
                    Bucket=settings.BUCKET_NAME,
                    Key=request.s3_key
                )
                content_type = s3_response.get(
                    'ContentType', 'application/octet-stream')
            except Exception as e:
                logger.error(f"S3 object not found: {str(e)}")
                raise ValueError("File not found in S3")

            # ファイル種別を判定
            is_image = content_type.startswith('image/')
            is_pdf = content_type == 'application/pdf' or request.filename.lower().endswith('.pdf')

            if is_image:
                # 画像ファイルの場合はリサイズ処理
                await self._handle_image_resize(request, content_type)

            # PDFファイルの場合は変換処理を開始
            if is_pdf:
                return await self._handle_pdf_conversion(request)
            else:
                # 画像ファイルの場合はそのまま処理待ちに
                update_image_status(request.image_id, "pending")
                return {
                    "status": "success",
                    "message": "Upload completed successfully",
                    "image_id": request.image_id,
                    "is_converting": False
                }

        except Exception as e:
            logger.error(f"Error handling upload complete: {str(e)}")
            raise

    async def _handle_image_resize(self, request: UploadCompleteRequest, content_type: str) -> None:
        """画像のリサイズ処理"""
        try:
            # S3から画像を取得
            s3_obj = s3_client.get_object(
                Bucket=settings.BUCKET_NAME,
                Key=request.s3_key
            )
            image_data = s3_obj['Body'].read()

            # 画像をリサイズ（resize_image関数が存在する場合）
            try:
                resized_image_data, was_resized, orig_size, new_size = resize_image(
                    image_data)

                if was_resized:
                    # リサイズされた画像をS3にアップロード
                    converted_s3_key = f"converted/{datetime.now().isoformat()}_{request.filename}"
                    s3_client.put_object(
                        Bucket=settings.BUCKET_NAME,
                        Key=converted_s3_key,
                        Body=resized_image_data,
                        ContentType=content_type
                    )
                    logger.info(f"リサイズ画像をアップロードしました: {converted_s3_key}")

                    # DynamoDBを更新
                    update_converted_image(
                        request.image_id,
                        converted_s3_key,
                        "pending",
                        orig_size,
                        new_size
                    )
                else:
                    logger.info("リサイズは不要です。元の画像を使用します。")
            except ImportError:
                logger.info(
                    "resize_image function not available, skipping resize")
        except Exception as e:
            logger.error(f"画像リサイズエラー: {str(e)}")
            # リサイズに失敗しても処理を続行

    async def _handle_pdf_conversion(self, request: UploadCompleteRequest) -> Dict[str, Any]:
        """PDF変換処理"""
        try:
            # ステータスを変換中に更新
            update_image_status(request.image_id, "converting")

            # バックグラウンドタスクとして変換処理を実行
            from main import background_task
            task_id = background_task.add_task(
                convert_pdf_to_image,
                request.image_id,
                request.s3_key
            )
            logger.info(
                f"Started PDF conversion task {task_id} for image {request.image_id}")

            return {
                "status": "success",
                "message": "Upload completed, PDF conversion started",
                "image_id": request.image_id,
                "is_converting": True
            }
        except Exception as e:
            logger.error(f"PDF conversion setup error: {str(e)}")
            raise

    async def get_image_stream(self, image_id: str) -> StreamingResponse:
        """画像をストリーミングで返す"""
        try:
            # 画像情報を取得
            image_data = get_image(image_id)
            if not image_data:
                raise ValueError("Image not found")

            s3_key = image_data.get("s3_key")
            if isinstance(s3_key, list):
                s3_key = s3_key[0]  # リストの場合は最初の要素

            # S3から画像を取得
            s3_response = s3_client.get_object(
                Bucket=self.bucket_name, Key=s3_key)
            image_data_bytes = s3_response['Body'].read()

            # Content-Typeを推定
            content_type = s3_response.get(
                'ContentType', 'application/octet-stream')

            # ストリーミングレスポンスを作成
            return StreamingResponse(
                io.BytesIO(image_data_bytes),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"inline; filename={image_data.get('filename', 'image')}"}
            )

        except Exception as e:
            logger.error(f"Error getting image stream: {str(e)}")
            raise

    async def generate_download_url(self, image_id: str) -> Dict[str, Any]:
        """ダウンロード用の署名付きURLを生成する（複数ページ対応）"""
        try:
            # 画像情報を取得
            image_data = get_image(image_id)
            if not image_data:
                raise ValueError("Image not found")

            # S3キーを抽出（リスト・文字列両対応）
            def extract_s3_keys_from_dynamo_data(dynamo_data):
                if isinstance(dynamo_data, list):
                    return dynamo_data
                elif isinstance(dynamo_data, str):
                    return [dynamo_data]
                return []

            converted_s3_keys = extract_s3_keys_from_dynamo_data(
                image_data.get("converted_s3_key"))
            s3_keys = extract_s3_keys_from_dynamo_data(
                image_data.get("s3_key"))

            # 使用するS3キーを決定
            if converted_s3_keys:
                # 変換後の画像がある場合
                target_s3_keys = converted_s3_keys
                bucket_name = self.bucket_name
                logger.info(f"変換後の画像のダウンロードURLを生成します: {bucket_name}")
            elif s3_keys:
                # 元画像を使用
                target_s3_keys = s3_keys
                bucket_name = self.bucket_name
                logger.info(f"元画像のダウンロードURLを生成します: {bucket_name}")
            else:
                raise ValueError("Image file not found")

            # 複数ページの署名付きURLを生成
            presigned_urls = []
            main_presigned_url = None
            main_content_type = 'application/octet-stream'

            for i, s3_key in enumerate(target_s3_keys):
                if not s3_key:
                    continue

                # S3オブジェクトのContent-Typeを取得
                try:
                    s3_response = s3_client.head_object(
                        Bucket=bucket_name,
                        Key=s3_key
                    )
                    content_type = s3_response.get(
                        'ContentType', 'application/octet-stream')
                except Exception:
                    content_type = 'application/octet-stream'

                # 署名付きURLの生成（有効期限は1時間）
                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': s3_key,
                        'ResponseContentType': content_type,
                        'ResponseCacheControl': 'no-cache'
                    },
                    ExpiresIn=3600,  # 1時間
                    HttpMethod='GET'
                )

                presigned_urls.append({
                    "page": i + 1,
                    "presigned_url": presigned_url,
                    "s3_key": s3_key
                })

                # 最初のページをメインとして設定
                if i == 0:
                    main_presigned_url = presigned_url
                    main_content_type = content_type

            if not presigned_urls:
                raise ValueError("No valid S3 keys found")

            logger.info(f"Generated download URL for image {image_id}")

            return {
                "presigned_url": main_presigned_url,  # 単一画像用のメインURL
                "presigned_urls": presigned_urls,
                "total_pages": len(presigned_urls),
                "is_multipage": len(presigned_urls) > 1,
                "content_type": main_content_type,
                "filename": image_data.get("filename"),
                "is_converted": bool(converted_s3_keys)
            }

        except Exception as e:
            logger.error(f"Error generating download URL: {str(e)}")
            raise

    async def get_images_list(self) -> Dict[str, Any]:
        """画像一覧を取得する"""
        try:
            # 全画像を取得
            images = get_images()

            # レスポンス形式に変換
            result = {
                "images": images,
                "total": len(images)
            }

            logger.info(f"Retrieved {len(images)} images")
            return result

        except Exception as e:
            logger.error(f"Error getting images list: {str(e)}")
            raise
