from clients import s3_client
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError

from config import settings
from repositories.schema_repository import get_app_schema
from repositories.image_repository import create_image_record, get_images_by_sync_source, get_existing_sync_sources
from schemas import UploadCompleteRequest
from services.upload_service import UploadService

logger = logging.getLogger(__name__)


class S3SyncService:
    """S3同期処理を管理するサービスクラス"""

    def __init__(self, upload_service=None):
        self.bucket_name = settings.BUCKET_NAME
        self.sync_bucket_name = settings.SYNC_BUCKET_NAME
        self._upload_service = upload_service

    async def sync_s3_files(self, app_name: str, prefix: Optional[str] = None) -> Dict[str, Any]:
        """S3バケットからファイルを同期する"""
        try:
            # アプリケーションの入力方法設定を取得
            app_schema = get_app_schema(app_name)
            if not app_schema:
                raise ValueError(f"アプリが見つかりません: {app_name}")

            input_methods = app_schema.get("input_methods", {})

            # S3同期が有効かチェック
            if not input_methods.get("s3_sync", False):
                raise ValueError(f"S3同期はこのアプリケーションでは有効になっていません: {app_name}")

            # 同期バケットからファイル一覧を取得
            s3_path = f"{app_name}/"
            if prefix:
                s3_path = f"{app_name}/{prefix}"

            files = await self._list_s3_files(self.sync_bucket_name, s3_path)

            # フォルダ構造を構築
            structure = self._build_folder_tree(files, app_name)

            return {
                "status": "success",
                "bucket_name": self.sync_bucket_name,
                "s3_path": s3_path,
                "structure": structure,
                "files": files,
                "total_files": len(files)
            }

        except Exception as e:
            logger.error(f"Error syncing S3 files: {str(e)}")
            raise

    async def import_s3_file(self, app_name: str, file_data: dict, uploaded_by: str = None) -> Dict[str, str]:
        """S3バケットからファイルをインポートしてOCR処理を開始する"""
        try:
            # アプリケーションの入力方法設定を取得
            app_schema = get_app_schema(app_name)
            if not app_schema:
                raise ValueError(f"アプリが見つかりません: {app_name}")

            input_methods = app_schema.get("input_methods", {})

            # S3同期が有効かチェック
            if not input_methods.get("s3_sync", False):
                raise ValueError(f"S3同期はこのアプリケーションでは有効になっていません: {app_name}")

            # ファイル情報を取得
            source_bucket = file_data.get("bucket")
            source_key = file_data.get("key")
            filename = file_data.get("filename")
            page_processing_mode = file_data.get("page_processing_mode", "combined")

            if not all([source_bucket, source_key, filename]):
                raise ValueError("bucket, key, filename are required")

            # 重複チェック
            existing_files = get_images_by_sync_source(filename, source_key, app_name)
            if existing_files:
                raise ValueError(f"ファイル '{filename}' は既にインポート済みです")

            # 新しい画像IDを生成
            image_id = str(uuid.uuid4())

            # コピー先のS3キーを生成
            destination_key = f"s3-imports/{datetime.now().isoformat()}_{filename}"

            # ファイルを自分のバケットにコピー
            await self._copy_s3_file(source_bucket, source_key, destination_key)

            # DynamoDBにレコードを作成
            create_image_record(
                image_id=image_id,
                filename=filename,
                s3_key=destination_key,
                app_name=app_name,
                status="uploading",
                page_processing_mode=page_processing_mode,
                sync_source_path=source_key,
                uploaded_by=uploaded_by
            )

            # アップロード完了処理を実行（OCR処理開始）
            upload_service = self._upload_service or UploadService()
            upload_request = UploadCompleteRequest(
                image_id=image_id,
                filename=filename,
                s3_key=destination_key,
                app_name=app_name,
                page_processing_mode=page_processing_mode
            )
            
            # 直接アップロードと同じOCR処理フローを実行
            processing_result = await upload_service.handle_upload_complete(upload_request)

            logger.info(f"Imported S3 file {source_key} as image {image_id} and started processing")

            return {
                "status": "success",
                "image_id": image_id,
                "message": f"File {filename} imported successfully"
            }

        except Exception as e:
            logger.error(f"Error importing S3 file: {str(e)}")
            raise

    async def get_files_with_duplicate_check(self, app_name: str, prefix: Optional[str] = None) -> Dict[str, Any]:
        """S3ファイル一覧を重複チェック付きで取得する"""
        try:
            # 基本のファイル一覧を取得
            sync_result = await self.sync_s3_files(app_name, prefix)
            files = sync_result.get("files", [])
            
            if not files:
                return sync_result
            
            # S3キーのリストを作成
            s3_keys = [file["key"] for file in files]
            
            # 重複チェックを実行
            existing_files = await self.check_existing_files(app_name, s3_keys)
            
            # ファイル情報に重複フラグを追加
            for file in files:
                file["is_existing"] = existing_files.get(file["key"], False)
            
            # 結果に重複情報を追加
            sync_result["files"] = files
            sync_result["duplicate_count"] = len([k for k, v in existing_files.items() if v])
            
            return sync_result
            
        except Exception as e:
            logger.error(f"重複チェック付きファイル一覧取得エラー: {str(e)}")
            raise

    async def check_existing_files(self, app_name: str, s3_keys: List[str]) -> Dict[str, bool]:
        """既存ファイルをバッチチェックする（1 回の scan で全件取得）"""
        try:
            existing_sources = get_existing_sync_sources(app_name)
            return {s3_key: s3_key in existing_sources for s3_key in s3_keys}
        except Exception as e:
            logger.error(f"既存ファイルチェックエラー: {str(e)}")
            raise

    def _build_folder_tree(self, files: List[Dict[str, Any]], app_name: str) -> Dict[str, Any]:
        """ファイル一覧からフォルダツリー構造を構築する"""
        tree = {}
        
        for file in files:
            # app_name/を除いた相対パスを取得
            full_key = file['key']
            if full_key.startswith(f"{app_name}/"):
                relative_path = full_key[len(f"{app_name}/"):]
            else:
                relative_path = full_key
            
            # パスを分割してツリー構造を構築
            path_parts = relative_path.split('/')
            current = tree
            
            # フォルダ部分を処理
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {"type": "folder", "children": {}}
                current = current[part]["children"]
            
            # ファイル部分を処理
            file_name = path_parts[-1]
            current[file_name] = {
                "type": "file", 
                "data": {
                    **file,
                    "relative_path": relative_path
                }
            }
        
        return tree

    async def _list_s3_files(self, bucket_name: str, prefix: str) -> List[Dict[str, Any]]:
        """S3バケットからファイル一覧を取得する"""
        try:
            files = []
            paginator = s3_client.get_paginator('list_objects_v2')

            page_iterator = paginator.paginate(
                Bucket=bucket_name,
                Prefix=prefix
            )

            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if not obj['Key'].endswith('/'):
                            files.append({
                                "key": obj['Key'],
                                "filename": obj['Key'].split('/')[-1],
                                "size": obj['Size'],
                                "last_modified": obj['LastModified'].isoformat(),
                                "bucket": bucket_name
                            })

            return files

        except ClientError as e:
            logger.error(f"Error listing S3 files: {str(e)}")
            raise ValueError(f"S3バケットへのアクセスに失敗しました: {str(e)}")

    async def _copy_s3_file(self, source_bucket: str, source_key: str, destination_key: str) -> None:
        """S3ファイルを自分のバケットにコピーする"""
        try:
            copy_source = {
                'Bucket': source_bucket,
                'Key': source_key
            }

            s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=destination_key
            )

            logger.info(f"Copied S3 file from {source_bucket}/{source_key} to {self.bucket_name}/{destination_key}")

        except ClientError as e:
            logger.error(f"Error copying S3 file: {str(e)}")
            raise ValueError(f"S3ファイルのコピーに失敗しました: {str(e)}")
