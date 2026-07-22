from clients import s3_client
import boto3
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError

from config import settings
from exceptions import NotFoundError, BadRequestError
from domains.image_status import ImageStatus
from repositories.schema_repository import get_app_schema
from repositories.image_repository import create_image_record, get_existing_sync_sources, update_image_status

logger = logging.getLogger(__name__)


class S3SyncService:
    """S3同期処理を管理するサービスクラス"""

    def __init__(self):
        self.bucket_name = settings.BUCKET_NAME
        self.sync_bucket_name = settings.SYNC_BUCKET_NAME

    async def sync_s3_files(self, app_name: str, prefix: Optional[str] = None) -> Dict[str, Any]:
        """S3バケットからファイルを同期する"""
        try:
            # アプリケーションの入力方法設定を取得
            app_schema = get_app_schema(app_name)
            if not app_schema:
                raise NotFoundError(f"アプリが見つかりません: {app_name}")

            input_methods = app_schema.get("input_methods", {})

            # S3同期が有効かチェック
            if not input_methods.get("s3_sync", False):
                raise BadRequestError(f"S3同期はこのアプリケーションでは有効になっていません: {app_name}")

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

    async def import_s3_files_batch(
        self, app_name: str, files: List[dict], page_processing_mode: str, uploaded_by: str = None
    ) -> Dict[str, Any]:
        """複数の S3 ファイルをまとめてインポートする。

        レコード作成（重複チェック込み）だけを同期で行い、S3 コピー・変換等の重い処理は
        S3SyncImport worker に async invoke で委譲する。ブラウザは応答後すぐ閉じてよい。
        """
        app_schema = get_app_schema(app_name)
        if not app_schema:
            raise NotFoundError(f"アプリが見つかりません: {app_name}")
        if not app_schema.get("input_methods", {}).get("s3_sync", False):
            raise BadRequestError(f"S3同期はこのアプリケーションでは有効になっていません: {app_name}")

        if not settings.S3_SYNC_IMPORT_FUNCTION_NAME:
            raise RuntimeError("S3_SYNC_IMPORT_FUNCTION_NAME is not configured")

        # 1 回の scan で既存の同期元パスを取得しバッチ重複チェック
        existing_sources = get_existing_sync_sources(app_name)

        items = []
        skipped = []
        created_image_ids = []
        for file_data in files:
            source_bucket = file_data.get("bucket")
            source_key = file_data.get("key")
            filename = file_data.get("filename")

            if not all([source_bucket, source_key, filename]):
                skipped.append({"key": source_key, "reason": "bucket/key/filename が不足"})
                continue
            if source_bucket != self.sync_bucket_name:
                skipped.append({"key": source_key, "reason": "無効なソースバケット"})
                continue
            if source_key in existing_sources:
                skipped.append({"key": source_key, "reason": "インポート済み"})
                continue

            image_id = str(uuid.uuid4())
            destination_key = f"s3-imports/{datetime.now().isoformat()}_{filename}"
            create_image_record(
                image_id=image_id,
                filename=filename,
                s3_key=destination_key,
                app_name=app_name,
                status=ImageStatus.UPLOADING,
                page_processing_mode=page_processing_mode,
                sync_source_path=source_key,
                uploaded_by=uploaded_by,
            )
            created_image_ids.append(image_id)
            items.append({
                "image_id": image_id,
                "source_bucket": source_bucket,
                "source_key": source_key,
                "destination_key": destination_key,
                "filename": filename,
            })

        if items:
            try:
                boto3.client("lambda").invoke(
                    FunctionName=settings.S3_SYNC_IMPORT_FUNCTION_NAME,
                    InvocationType="Event",
                    Payload=json.dumps({
                        "app_name": app_name,
                        "page_processing_mode": page_processing_mode,
                        "items": items,
                    }),
                )
            except Exception:
                # invoke 失敗時は作成済みレコードを FAILED にし、UPLOADING のまま残さない
                for image_id in created_image_ids:
                    try:
                        update_image_status(image_id, ImageStatus.FAILED)
                    except Exception as update_err:
                        logger.error(f"Failed to mark image {image_id} FAILED: {update_err}")
                logger.error("Failed to invoke S3SyncImport worker")
                raise

        logger.info(f"Accepted S3 import batch for {app_name}: {len(items)} queued, {len(skipped)} skipped")
        return {
            "status": "accepted",
            "imported_count": len(items),
            "image_ids": created_image_ids,
            "skipped": skipped,
        }

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
            raise RuntimeError(f"S3バケットへのアクセスに失敗しました: {str(e)}")
