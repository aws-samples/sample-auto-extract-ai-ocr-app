from clients import s3_client
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from repositories import (
    create_image_record, get_image, get_images, update_image_status, update_converted_image,
    get_children_by_parent_id, delete_image as repo_delete_image
)
from schemas import (
    PresignedUrlRequest, PresignedUrlResponse, UploadCompleteRequest
)
from config import settings
from utils import resize_image, decimal_to_float
from repositories import get_app_schemas, get_app_input_methods
from repositories.usecase_repository import get_permitted_app_names
from repositories import user_repository
from background import BackgroundTaskExtension
from services.pdf_conversion_service import convert_pdf_to_image
from schemas.image import ImageInfo

logger = logging.getLogger(__name__)

# 共通のS3クライアントを使用


class UploadService:
    """アップロード処理を管理するサービスクラス"""

    def __init__(self, background_task: Optional[BackgroundTaskExtension] = None):
        self.bucket_name = settings.BUCKET_NAME
        self.background_task = background_task

    async def generate_presigned_url(self, request: PresignedUrlRequest, uploaded_by: str = None) -> PresignedUrlResponse:
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
                logger.error(f"Invalid app name: {request.app_name}")
                raise ValueError(f"Invalid app name: {request.app_name}")

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
                page_processing_mode=request.page_processing_mode,  # 追加
                uploaded_by=uploaded_by
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
                    # リサイズ不要でも元の画像をconverted_s3_keyとして設定
                    update_converted_image(
                        request.image_id,
                        request.s3_key,
                        "pending",
                        orig_size,
                        orig_size
                    )
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
            if not self.background_task:
                raise ValueError("background_task is not configured for PDF conversion")
            task_id = self.background_task.add_task(
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

    async def get_image_stream(self, image_id: str) -> Tuple[bytes, str, str]:
        """画像データを返す

        Returns:
            (image_bytes, content_type, filename) のタプル
        """
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

            content_type = s3_response.get(
                'ContentType', 'application/octet-stream')
            filename = image_data.get('filename', 'image')

            return (image_data_bytes, content_type, filename)

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

    @staticmethod
    def _serialize_images(images: list[dict]) -> list[dict]:
        """DynamoDB の画像レコードを API レスポンス形式（camelCase）に変換する"""
        result = []
        for img in images:
            try:
                # DynamoDB の Decimal 型を Python の int/float に変換
                converted = decimal_to_float(img)
                info = ImageInfo.model_validate(converted)
                result.append(info.model_dump(by_alias=True))
            except Exception as e:
                logger.error(f"Image serialization error for {img.get('id', '?')}: {e}; raw_keys={sorted(img.keys())}")
                result.append({"id": img.get("id", ""), "name": img.get("filename", ""), "status": img.get("status", "")})
        return result

    async def get_images_list(self, app_name: str = None, uploaded_by: str = None) -> Dict[str, Any]:
        """画像一覧を取得する"""
        try:
            images = get_images(app_name, uploaded_by=uploaded_by)
            self._enrich_uploaded_by_email(images)

            serialized = self._serialize_images(images)
            result = {
                "images": serialized,
                "total": len(serialized)
            }

            logger.info(f"Retrieved {len(serialized)} images")
            return result

        except Exception as e:
            logger.error(f"Error getting images list: {str(e)}")
            raise

    async def get_images_for_user(self, user_id: str, role: str, app_name: str = None) -> Dict[str, Any]:
        """ユーザーの権限に応じた画像一覧を取得する

        Args:
            user_id: ユーザーID
            role: システムロール（admin / author / reader）
            app_name: ユースケースでフィルタする場合に指定
        """
        if role == "admin":
            return await self.get_images_list(app_name)

        permitted = get_permitted_app_names(user_id)
        if not permitted:
            return {"images": [], "total": 0}

        return await self.get_images_for_permitted_apps(permitted, app_name_filter=app_name)

    async def get_images_for_permitted_apps(self, app_names: list[str], app_name_filter: str = None) -> Dict[str, Any]:
        """権限のあるユースケースの画像一覧を取得する

        Args:
            app_names: ユーザーが権限を持つ app_name のリスト
            app_name_filter: 特定のユースケースでさらに絞り込む場合に指定
        """
        try:
            if app_name_filter:
                # フィルタ指定時は権限チェック済みの app_name のみ取得
                if app_name_filter not in app_names:
                    return {"images": [], "total": 0}
                target_apps = [app_name_filter]
            else:
                target_apps = app_names

            all_images = []
            for name in target_apps:
                images = get_images(app_name=name)
                all_images.extend(images)

            # upload_time 降順でソート
            all_images.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
            self._enrich_uploaded_by_email(all_images)

            serialized = self._serialize_images(all_images)
            return {
                "images": serialized,
                "total": len(serialized)
            }
        except Exception as e:
            logger.error(f"Error getting images for permitted apps: {str(e)}")
            raise

    @staticmethod
    def _enrich_uploaded_by_email(images: list[dict]) -> None:
        """画像リストに uploaded_by_email / verified_by_email を付与する"""
        subs = set()
        for img in images:
            if img.get("uploaded_by"):
                subs.add(img["uploaded_by"])
            if img.get("verified_by"):
                subs.add(img["verified_by"])
        if not subs:
            return
        email_map = user_repository.get_emails_by_cognito_subs(subs)
        for img in images:
            img["uploaded_by_email"] = email_map.get(img.get("uploaded_by", ""), "")
            img["verified_by_email"] = email_map.get(img.get("verified_by", ""), "")

    async def delete_image(self, image_id: str, cognito_sub: str = None, is_admin: bool = False) -> Dict[str, Any]:
        """画像を削除する

        Args:
            image_id: 削除対象の画像ID
            cognito_sub: 操作ユーザーの cognito_sub（所有者チェック用）
            is_admin: admin ロールの場合 True（所有者チェックをスキップ）
        """
        try:
            image = get_image(image_id)
            if not image:
                raise ValueError("Image not found")

            # 所有者チェック（admin 以外）
            if not is_admin:
                if not cognito_sub or image.get("uploaded_by") != cognito_sub:
                    raise PermissionError("Forbidden: not the owner")
            
            parent_document_id = image.get("parent_document_id")
            page_processing_mode = image.get("page_processing_mode")
            total_pages = image.get("total_pages", 0)
            
            # 親ファイルの場合（個別処理で2ページ以上、parent_document_idなし）
            is_parent = (not parent_document_id and 
                        page_processing_mode == "individual" and 
                        total_pages > 1)
            
            if is_parent:
                # 親ファイルの場合、全ての子ファイルも削除
                children = get_children_by_parent_id(image_id)
                for child in children:
                    repo_delete_image(child['id'])
                    logger.info(f"Deleted child image: {child['id']}")
            
            # 子ファイルの場合、削除前に残りの子ファイル数をチェック
            remaining_count = 0
            if parent_document_id:
                all_children = get_children_by_parent_id(parent_document_id)
                # 削除前の子ファイル数をカウント（自分自身を除く）
                remaining_count = len([c for c in all_children if c['id'] != image_id])
                logger.info(f"Remaining children count (before deletion): {remaining_count}")
            
            # 対象ファイルを削除
            repo_delete_image(image_id)
            logger.info(f"Deleted image: {image_id}")
            
            # 子ファイルの場合、残りが0なら親も削除
            if parent_document_id and remaining_count == 0:
                # 子ファイルが全て削除されたら親も削除
                repo_delete_image(parent_document_id)
                logger.info(f"Deleted parent image: {parent_document_id}")
            
            return {"status": "success", "message": "Image deleted successfully"}

        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}")
            raise
