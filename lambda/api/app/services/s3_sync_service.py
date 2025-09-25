from clients import s3_client
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError

from config import settings
from app_schema import get_app_input_methods
from database import create_image_record

logger = logging.getLogger(__name__)


class S3SyncService:
    """S3同期処理を管理するサービスクラス"""

    def __init__(self):
        self.bucket_name = settings.BUCKET_NAME

    async def sync_s3_files(self, app_name: str, prefix: Optional[str] = None) -> Dict[str, Any]:
        """S3バケットからファイルを同期する"""
        try:
            # アプリケーションの入力方法設定を取得
            input_methods = get_app_input_methods(app_name)

            # S3同期が有効かチェック
            if not input_methods.get("s3_sync", False):
                raise ValueError(f"S3同期はこのアプリケーションでは有効になっていません: {app_name}")

            # S3 URIを取得
            s3_uri = input_methods.get("s3_uri", "")

            if not s3_uri:
                raise ValueError(f"S3 URIが設定されていません: {app_name}")

            # S3 URIを解析（s3://bucket-name/path/to/folder/）
            if not s3_uri.startswith("s3://"):
                raise ValueError(f"無効なS3 URI形式です: {s3_uri}")

            # s3://を削除
            s3_uri_without_prefix = s3_uri[5:]

            # バケット名とパスに分割
            parts = s3_uri_without_prefix.split('/', 1)
            bucket_name = parts[0]
            # パスが指定されていない場合は空文字列をデフォルトとする
            s3_path = parts[1] if len(parts) > 1 else ""

            # 指定されたプレフィックスがある場合は使用
            if prefix:
                s3_path = prefix

            # S3からファイル一覧を取得
            files = await self._list_s3_files(bucket_name, s3_path)

            logger.info(
                f"Found {len(files)} files in S3 bucket {bucket_name}/{s3_path}")

            return {
                "status": "success",
                "bucket_name": bucket_name,
                "s3_path": s3_path,
                "files": files,
                "total_files": len(files)
            }

        except Exception as e:
            logger.error(f"Error syncing S3 files: {str(e)}")
            raise

    async def import_s3_file(self, app_name: str, file_data: dict) -> Dict[str, str]:
        """S3バケットからファイルをインポートしてOCR処理を開始する"""
        try:
            # アプリケーションの入力方法設定を取得
            input_methods = get_app_input_methods(app_name)

            # S3同期が有効かチェック
            if not input_methods.get("s3_sync", False):
                raise ValueError(f"S3同期はこのアプリケーションでは有効になっていません: {app_name}")

            # S3 URIを取得
            s3_uri = input_methods.get("s3_uri", "")

            if not s3_uri:
                raise ValueError(f"S3 URIが設定されていません: {app_name}")

            # ファイル情報を取得
            source_bucket = file_data.get("bucket")
            source_key = file_data.get("key")
            filename = file_data.get("filename")

            if not all([source_bucket, source_key, filename]):
                raise ValueError("bucket, key, filename are required")

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
                status="uploaded"
            )

            logger.info(f"Imported S3 file {source_key} as image {image_id}")

            return {
                "status": "success",
                "image_id": image_id,
                "message": f"File {filename} imported successfully"
            }

        except Exception as e:
            logger.error(f"Error importing S3 file: {str(e)}")
            raise

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
                        # ファイルのみを対象とする（フォルダは除外）
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

            logger.info(
                f"Copied S3 file from {source_bucket}/{source_key} to {self.bucket_name}/{destination_key}")

        except ClientError as e:
            logger.error(f"Error copying S3 file: {str(e)}")
            raise ValueError(f"S3ファイルのコピーに失敗しました: {str(e)}")
