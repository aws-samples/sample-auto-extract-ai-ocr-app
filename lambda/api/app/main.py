from botocore.config import Config
from background import BackgroundTaskExtension
import sys
from pydantic import BaseModel
import time
import base64
import logging
import uuid
from datetime import datetime, timedelta
from ocr import perform_ocr, process_information_extraction_with_ocr, process_information_extraction_without_ocr
from app_schema import (
    get_extraction_fields_for_app,
    get_field_names_for_app,
    get_app_display_name,
    DEFAULT_APP,
    get_app_schemas,
    get_app_input_methods,
    update_app_schema,
    delete_app_schema
)
from database import (
    get_db_info, get_db_version, create_image_record, get_images,
    update_image_status, update_ocr_result, update_extracted_info,
    get_image, create_job, update_job_status, get_job, get_images_by_job_id, update_converted_image,
    delete_images_by_app_name, delete_jobs_by_app_name
)
import io
from PIL import Image
import fitz
from io import BytesIO
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import re
import cv2
import numpy as np
import tempfile
import json
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from decimal import Decimal

# DynamoDB Decimal型をJSON serializable に変換するヘルパー関数


def decimal_to_float(obj):
    """Decimal型をfloat型に変換してJSON serializable にする"""
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


# OCR有効フラグを取得
ENABLE_OCR = os.environ.get('ENABLE_OCR', 'true').lower() == 'true'

# バックグラウンドタスク拡張機能をインポート

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# バックグラウンドタスク拡張機能を初期化
background_task = BackgroundTaskExtension()

# CORS 設定
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# リクエスト完了時にバックグラウンドタスクに通知するミドルウェア
@app.middleware("http")
async def send_done_message(request, call_next):
    response = await call_next(request)
    background_task.done()
    return response

# boto3 クライアントの作成
s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))


# リクエスト/レスポンスモデルの定義
class PresignedUrlRequest(BaseModel):
    filename: str
    content_type: str
    app_name: str = DEFAULT_APP


class PresignedUrlResponse(BaseModel):
    presigned_url: str
    s3_key: str
    image_id: str


class UploadCompleteRequest(BaseModel):
    image_id: str
    filename: str
    s3_key: str
    app_name: str = DEFAULT_APP


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/apps")
async def get_apps():
    """利用可能なアプリケーション一覧を返す"""
    return get_app_schemas()


@app.get("/apps/{app_name}")
async def get_app_details(app_name: str):
    """特定のアプリケーションの詳細を返す"""
    app_schemas = get_app_schemas()
    for app in app_schemas.get("apps", []):
        if app["name"] == app_name:
            return app
    raise HTTPException(status_code=404, detail=f"App '{app_name}' not found")


@app.get("/apps/{app_name}/fields")
async def get_app_fields(app_name: str):
    """特定のアプリケーションの抽出フィールド定義を返す"""
    return get_extraction_fields_for_app(app_name)


@app.get("/apps/{app_name}/custom-prompt")
async def get_app_custom_prompt(app_name: str):
    """特定のアプリケーションのカスタムプロンプトを取得"""
    from app_schema import get_custom_prompt_for_app
    custom_prompt = get_custom_prompt_for_app(app_name)
    return {"custom_prompt": custom_prompt}


@app.put("/apps/{app_name}/custom-prompt")
async def update_app_custom_prompt(app_name: str, data: dict = Body(...)):
    """特定のアプリケーションのカスタムプロンプトを更新"""
    try:
        # アプリが存在するか確認
        app_schemas = get_app_schemas()
        app_data = None
        for app in app_schemas.get("apps", []):
            if app["name"] == app_name:
                app_data = app
                break

        if not app_data:
            raise HTTPException(
                status_code=404, detail=f"App '{app_name}' not found")

        # カスタムプロンプトを更新
        app_data["custom_prompt"] = data.get("custom_prompt", "")

        # スキーマを更新
        success = update_app_schema(app_name, app_data)

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to update custom prompt")

        return {"message": "Custom prompt updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating custom prompt: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error updating custom prompt: {str(e)}")


@app.post("/generate-presigned-url")
async def generate_presigned_url(request: PresignedUrlRequest):
    """署名付きURLを生成して返す"""
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
            raise HTTPException(
                status_code=400, detail=f"ファイルアップロードはこのアプリケーションでは無効です: {request.app_name}")

        # 一意のS3キーを生成
        image_id = str(uuid.uuid4())
        s3_key = f"uploads/{datetime.now().isoformat()}_{request.filename}"

        # 署名付きURLの生成（有効期限は15分）
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': os.getenv("BUCKET_NAME"),
                'Key': s3_key,
                'ContentType': request.content_type
            },
            ExpiresIn=900,  # 15分
            HttpMethod='PUT'
        )

        # DynamoDBに画像レコードを作成（ステータスはuploading）
        create_image_record(
            image_id=image_id,
            filename=request.filename,
            s3_key=s3_key,
            app_name=request.app_name,
            status="uploading"  # 新しいステータス
        )

        return PresignedUrlResponse(
            presigned_url=presigned_url,
            s3_key=s3_key,
            image_id=image_id
        )

    except NoCredentialsError:
        raise HTTPException(
            status_code=500, detail="S3 credentials not available")
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating presigned URL: {str(e)}")


@app.post("/upload-complete")
async def upload_complete(request: UploadCompleteRequest):
    """クライアントからのアップロード完了通知を処理"""
    try:
        # S3オブジェクトの存在確認
        try:
            s3_response = s3_client.head_object(
                Bucket=os.getenv("BUCKET_NAME"),
                Key=request.s3_key
            )
            content_type = s3_response.get(
                'ContentType', 'application/octet-stream')
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise HTTPException(
                    status_code=404, detail=f"File not found in S3: {request.s3_key}")
            else:
                raise

        # 画像ファイルの場合はリサイズ処理を行う
        is_image = content_type.startswith('image/')
        is_pdf = content_type == 'application/pdf' or request.filename.lower().endswith('.pdf')

        if is_image:
            try:
                # S3から画像を取得
                s3_obj = s3_client.get_object(
                    Bucket=os.getenv("BUCKET_NAME"),
                    Key=request.s3_key
                )
                image_data = s3_obj['Body'].read()

                # 画像をリサイズ
                from utils import resize_image
                resized_image_data, was_resized, orig_size, new_size = resize_image(
                    image_data)

                if was_resized:
                    # リサイズされた画像をS3にアップロード
                    converted_s3_key = f"converted/{datetime.now().isoformat()}_{request.filename}"
                    s3_client.put_object(
                        Bucket=os.getenv("BUCKET_NAME"),
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
            except Exception as e:
                logger.error(f"画像リサイズエラー: {str(e)}")
                # リサイズに失敗しても処理を続行

        # PDFファイルの場合は変換処理を開始
        if is_pdf:
            # ステータスを変換中に更新
            update_image_status(request.image_id, "converting")

            # バックグラウンドタスクとして変換処理を実行
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
        else:
            # 画像ファイルの場合はそのまま処理待ちに
            update_image_status(request.image_id, "pending")

            return {
                "status": "success",
                "message": "Upload completed successfully",
                "image_id": request.image_id,
                "is_converting": False
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload completion: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing upload completion: {str(e)}")


@app.get("/images")
async def get_images_endpoint(app_name: Optional[str] = None):
    try:
        # DynamoDB から画像一覧を取得
        images_list = get_images(app_name)
        return {"images": images_list}
    except Exception as e:
        logger.error(f"画像一覧取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


@app.post("/s3-sync/{app_name}")
async def sync_s3_files(app_name: str, prefix: Optional[str] = None):
    """S3バケットからファイルを同期する"""
    try:
        # アプリケーションの入力方法設定を取得
        input_methods = get_app_input_methods(app_name)

        # S3同期が有効かチェック
        if not input_methods.get("s3_sync", False):
            raise HTTPException(
                status_code=400, detail=f"S3同期はこのアプリケーションでは有効になっていません: {app_name}")

        # S3 URIを取得
        s3_uri = input_methods.get("s3_uri", "")

        if not s3_uri:
            raise HTTPException(
                status_code=400, detail=f"S3 URIが設定されていません: {app_name}")

        # S3 URIを解析（s3://bucket-name/path/to/folder/）
        if not s3_uri.startswith("s3://"):
            raise HTTPException(
                status_code=400, detail=f"無効なS3 URI形式です: {s3_uri}")

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

        logger.info(f"S3同期を実行します: バケット={bucket_name}, パス={s3_path}")

        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=s3_path
            )

            files = []
            if "Contents" in response:
                for item in response["Contents"]:
                    # ディレクトリは除外
                    if not item["Key"].endswith("/"):
                        files.append({
                            "key": item["Key"],
                            "size": item["Size"],
                            "last_modified": item["LastModified"].isoformat(),
                            "filename": os.path.basename(item["Key"])
                        })

            return {
                "app_name": app_name,
                "bucket": bucket_name,
                "prefix": s3_path,
                "files": files
            }

        except ClientError as e:
            logger.error(f"S3ファイル一覧取得エラー: {str(e)}")
            raise HTTPException(status_code=500, detail=f"S3エラー: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"S3同期エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"S3同期エラー: {str(e)}")


@app.post("/s3-sync/{app_name}/import")
async def import_s3_file(app_name: str, file_data: dict):
    """S3バケットからファイルをインポートしてOCR処理を開始する"""
    try:
        # アプリケーションの入力方法設定を取得
        input_methods = get_app_input_methods(app_name)

        # S3同期が有効かチェック
        if not input_methods.get("s3_sync", False):
            raise HTTPException(
                status_code=400, detail=f"S3同期はこのアプリケーションでは有効になっていません: {app_name}")

        # S3 URIを取得
        s3_uri = input_methods.get("s3_uri", "")

        if not s3_uri:
            raise HTTPException(
                status_code=400, detail=f"S3 URIが設定されていません: {app_name}")

        # S3 URIを解析（s3://bucket-name/path/to/folder/）
        if not s3_uri.startswith("s3://"):
            raise HTTPException(
                status_code=400, detail=f"無効なS3 URI形式です: {s3_uri}")

        # s3://を削除
        s3_uri_without_prefix = s3_uri[5:]

        # 最初の/でバケット名とパスに分割
        parts = s3_uri_without_prefix.split('/', 1)
        bucket_name = parts[0]
        # パスが指定されていない場合は空文字列をデフォルトとする
        s3_path = parts[1] if len(parts) > 1 else ""

        # ファイルキーを取得
        s3_key = file_data.get("key")
        if not s3_key:
            raise HTTPException(status_code=400, detail="ファイルキーが指定されていません")

        # ファイル名を取得
        filename = file_data.get("filename") or os.path.basename(s3_key)

        # S3オブジェクトの存在確認
        try:
            s3_response = s3_client.head_object(
                Bucket=bucket_name,
                Key=s3_key
            )
            content_type = s3_response.get(
                'ContentType', 'application/octet-stream')
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise HTTPException(
                    status_code=404, detail=f"ファイルがS3に見つかりません: {bucket_name}/{s3_key}")
            else:
                raise

        # 画像レコードを作成
        image_id = str(uuid.uuid4())
        create_image_record(
            image_id=image_id,
            filename=filename,
            s3_key=s3_key,
            app_name=app_name,
            status="pending"
        )

        # PDFファイルの場合は変換処理を開始
        is_pdf = content_type == 'application/pdf' or filename.lower().endswith('.pdf')
        if is_pdf:
            # ステータスを変換中に更新
            update_image_status(image_id, "converting")

            # バックグラウンドタスクとして変換処理を実行
            task_id = background_task.add_task(
                convert_pdf_to_image,
                image_id,
                s3_key
            )
            logger.info(
                f"Started PDF conversion task {task_id} for image {image_id}")

            return {
                "status": "success",
                "message": "S3ファイルのインポートが完了し、PDF変換を開始しました",
                "image_id": image_id,
                "is_converting": True
            }
        else:
            # 画像ファイルの場合はそのまま処理待ちに
            return {
                "status": "success",
                "message": "S3ファイルのインポートが完了しました",
                "image_id": image_id,
                "is_converting": False
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"S3ファイルインポートエラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"S3ファイルインポートエラー: {str(e)}")

        # ファイルキーを取得
        s3_key = file_data.get("key")
        if not s3_key:
            raise HTTPException(status_code=400, detail="ファイルキーが指定されていません")

        # ファイル名を取得
        filename = file_data.get("filename") or os.path.basename(s3_key)

        # S3オブジェクトの存在確認
        bucket_name = os.getenv("BUCKET_NAME")
        try:
            s3_response = s3_client.head_object(
                Bucket=bucket_name,
                Key=s3_key
            )
            content_type = s3_response.get(
                'ContentType', 'application/octet-stream')
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise HTTPException(
                    status_code=404, detail=f"ファイルがS3に見つかりません: {s3_key}")
            else:
                raise

        # 画像レコードを作成
        image_id = str(uuid.uuid4())
        create_image_record(
            image_id=image_id,
            filename=filename,
            s3_key=s3_key,
            app_name=app_name,
            status="pending"
        )

        # PDFファイルの場合は変換処理を開始
        is_pdf = content_type == 'application/pdf' or filename.lower().endswith('.pdf')
        if is_pdf:
            # ステータスを変換中に更新
            update_image_status(image_id, "converting")

            # バックグラウンドタスクとして変換処理を実行
            task_id = background_task.add_task(
                convert_pdf_to_image,
                image_id,
                s3_key
            )
            logger.info(
                f"Started PDF conversion task {task_id} for image {image_id}")

            return {
                "status": "success",
                "message": "S3ファイルのインポートが完了し、PDF変換を開始しました",
                "image_id": image_id,
                "is_converting": True
            }
        else:
            # 画像ファイルの場合はそのまま処理待ちに
            return {
                "status": "success",
                "message": "S3ファイルのインポートが完了しました",
                "image_id": image_id,
                "is_converting": False
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"S3ファイルインポートエラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"S3ファイルインポートエラー: {str(e)}")

# ここに残りのエンドポイントを追加します
# 残りのコードは次のステップで追加します

# Lambda Web Adapter用のハンドラー
handler = app


def convert_pdf_to_image(image_id: str, s3_key: str):
    """
    PDFを画像に変換し、S3にアップロードする

    Args:
        image_id (str): 画像ID
        s3_key (str): PDFファイルのS3キー
    """
    try:
        logger.info(f"PDFの変換を開始します: {image_id}, {s3_key}")

        # 画像情報を取得してバケット名を決定
        image_data = get_image(image_id)
        app_name = image_data.get("app_name", DEFAULT_APP)
        input_methods = get_app_input_methods(app_name)

        # S3 URIからバケット名を取得
        bucket_name = os.getenv("BUCKET_NAME")  # デフォルトバケット
        if input_methods.get("s3_sync", False) and input_methods.get("s3_uri"):
            s3_uri = input_methods["s3_uri"]
            if s3_uri.startswith("s3://"):
                parts = s3_uri[5:].split('/', 1)
                if len(parts) > 0:
                    bucket_name = parts[0]

        logger.info(f"S3バケット名: {bucket_name}")

        # S3から元ファイルを取得
        s3_response = s3_client.get_object(
            Bucket=bucket_name,  # S3 URIから取得したバケット名
            Key=s3_key
        )
        file_content = s3_response['Body'].read()

        # PDFを画像に変換
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(file_content)
            temp_pdf_path = temp_pdf.name

        try:
            # PDFを開く
            pdf_document = fitz.open(temp_pdf_path)

            if pdf_document.page_count == 0:
                raise ValueError("PDF has no pages")

            # 変換後のファイルは常に環境変数で指定されたバケットに保存
            upload_bucket = os.getenv("BUCKET_NAME")
            if not upload_bucket:
                raise ValueError("BUCKET_NAME environment variable is not set")

            logger.info(f"変換後のファイルの保存先バケット: {upload_bucket}")

            # 画像の結合が必要な場合（複数ページ）
            if pdf_document.page_count > 1:
                logger.info(f"複数ページPDFを処理します: {pdf_document.page_count}ページ")

                # 各ページを画像として処理
                page_images = []
                total_width = 0
                max_height = 0

                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(dpi=300)  # 高解像度で画像化

                    # PILイメージに変換
                    img = Image.frombytes(
                        "RGB", [pix.width, pix.height], pix.samples)
                    page_images.append(img)

                    # 幅と高さの計算用
                    total_width = max(total_width, img.width)
                    max_height += img.height

                # 結合画像の作成
                combined_image = Image.new('RGB', (total_width, max_height))
                y_offset = 0

                for img in page_images:
                    combined_image.paste(img, (0, y_offset))
                    y_offset += img.height

                # 結合画像をバイトストリームに変換
                img_byte_arr = io.BytesIO()
                combined_image.save(img_byte_arr, format='JPEG', quality=95)
                img_byte_arr.seek(0)

                # 元のサイズを記録
                original_size = (combined_image.width, combined_image.height)

                # 画像をリサイズ
                from utils import resize_image
                img_data = img_byte_arr.getvalue()
                resized_image_data, was_resized, orig_size, new_size = resize_image(
                    img_data)

                # 変換後のS3キーを生成
                filename_base = os.path.splitext(os.path.basename(s3_key))[0]
                converted_s3_key = f"converted/{datetime.now().isoformat()}_{filename_base}.jpg"

                # S3にアップロード（常に環境変数のバケットを使用）
                s3_client.put_object(
                    Bucket=upload_bucket,
                    Key=converted_s3_key,
                    Body=resized_image_data if was_resized else img_data,
                    ContentType='image/jpeg'
                )

            else:  # 単一ページの場合
                logger.info("単一ページPDFを処理します")

                # ページを画像として処理
                page = pdf_document[0]
                pix = page.get_pixmap(dpi=300)  # 高解像度で画像化

                # バイトデータをBytesIOオブジェクトに変換
                img_byte_arr = io.BytesIO()
                img = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples)
                img.save(img_byte_arr, format='JPEG', quality=95)
                img_byte_arr.seek(0)

                # 元のサイズを記録
                original_size = (img.width, img.height)

                # 画像をリサイズ
                from utils import resize_image
                img_data = img_byte_arr.getvalue()
                resized_image_data, was_resized, orig_size, new_size = resize_image(
                    img_data)

                # 変換後のS3キーを生成
                filename_base = os.path.splitext(os.path.basename(s3_key))[0]
                converted_s3_key = f"converted/{datetime.now().isoformat()}_{filename_base}.jpg"

                # S3にアップロード（常に環境変数のバケットを使用）
                s3_client.put_object(
                    Bucket=upload_bucket,
                    Key=converted_s3_key,
                    Body=resized_image_data if was_resized else img_data,
                    ContentType='image/jpeg'
                )

            # DynamoDBを更新
            update_converted_image(
                image_id,
                converted_s3_key,
                "pending",
                orig_size if was_resized else original_size,
                new_size if was_resized else original_size
            )
            logger.info(f"PDF変換が完了しました: {image_id}, {converted_s3_key}")

            # PDFを閉じる
            pdf_document.close()

        finally:
            # 一時ファイルの削除
            try:
                os.unlink(temp_pdf_path)
            except Exception as e:
                logger.warning(f"一時ファイルの削除に失敗しました: {str(e)}")

    except Exception as e:
        logger.error(f"PDF変換エラー: {str(e)}")
        update_image_status(image_id, "failed")
        # エラー情報も保存
        error_result = {
            "error": f"PDF変換エラー: {str(e)}",
            "words": []
        }
        try:
            update_ocr_result(image_id, error_result, "failed")
        except Exception as db_error:
            logger.error(f"エラー情報の保存に失敗しました: {str(db_error)}")


@app.post("/ocr/start")
async def start_ocr():
    job_id = str(uuid.uuid4())

    try:
        # ジョブを作成
        create_job(job_id, 'processing')

        # 保留中の画像のステータスを更新
        images_list = get_images()
        processing_images = []

        for image in images_list:
            if image.get("status") == "pending":
                update_image_status(image.get("id"), "processing", job_id)
                processing_images.append(image)

        # バックグラウンドタスクとしてOCR処理を実行
        if processing_images:
            logger.info(
                f"バックグラウンドタスクを開始します: job_id={job_id}, images={len(processing_images)}")
            task_id = background_task.add_task(process_ocr_job, job_id)
            logger.info(f"Started OCR job {job_id} with task ID {task_id}")
        else:
            logger.warning(f"処理対象の画像がありません: job_id={job_id}")

        return {"jobId": job_id}
    except Exception as e:
        logger.error(f"OCRジョブの開始エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def process_ocr_job(job_id: str):
    try:
        logger.info(f"バックグラウンドタスク開始: job_id={job_id}")
        # ジョブに関連する画像を取得
        images = get_images_by_job_id(job_id)
        logger.info(f"Processing job {job_id} with {len(images)} images")

        # 同時処理数を制限（例: 最大2枚ずつ処理）
        batch_size = 2
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1} with {len(batch)} images")

            for image in batch:
                image_id = image.get("id")

                try:
                    # 画像情報を取得
                    image_data = get_image(image_id)
                    s3_key = image_data.get("s3_key")

                    logger.info(
                        f"Processing image {image_id} with S3 key {s3_key}")

                    # S3 から画像をダウンロード
                    s3_response = s3_client.get_object(
                        Bucket=os.getenv("BUCKET_NAME"), Key=s3_key)
                    image_data = s3_response['Body'].read()

                    # OCRモードに応じて処理を分岐
                    if ENABLE_OCR:
                        # OCRありモード: 既存の処理
                        logger.info(f"OCRありモードで処理開始: {image_id}")

                        # OCR処理を行う（ocr.pyの関数を使用）
                        ocr_result = perform_ocr(image_data)

                        # OCR結果にエラーがある場合の処理
                        if "error" in ocr_result:
                            logger.error(
                                f"OCR処理でエラーが発生: {ocr_result['error']}")
                            update_image_status(image_id, "failed")
                            continue

                        logger.info(
                            f"Successfully processed {len(ocr_result.get('words', []))} words for image {image_id}")

                        # OCR結果をテキストとして取得（新しい構造に対応）
                        ocr_text = ocr_result.get("text", "")
                        if not ocr_text:
                            # フォールバック: wordsから結合
                            ocr_text = "\n".join([word.get("content", "")
                                                 for word in ocr_result.get("words", [])])

                        # DynamoDBにOCR結果を保存
                        logger.info(f"Saving OCR results for image {image_id}")
                        update_ocr_result(image_id, ocr_result, "processing")

                        # 情報抽出プロセスを実行
                        logger.info(
                            f"Starting information extraction for image {image_id}")

                        try:
                            # 情報抽出の実行
                            process_information_extraction_with_ocr(
                                image_id, ocr_result, ocr_text)

                            # 処理が完了したらOCRのステータスも完了に設定
                            update_image_status(image_id, "completed")
                            logger.info(
                                f"Successfully completed OCR and extraction for image {image_id}")
                        except Exception as extraction_error:
                            logger.error(
                                f"Error during information extraction: {str(extraction_error)}")
                            update_image_status(image_id, "failed")
                    else:
                        # OCRなしモード: 直接情報抽出
                        logger.info(f"OCRなしモードで処理開始: {image_id}")

                        try:
                            # OCRなしで直接情報抽出
                            process_information_extraction_without_ocr(
                                image_id)

                            update_image_status(image_id, "completed")
                            logger.info(
                                f"Successfully completed extraction without OCR for image {image_id}")
                        except Exception as extraction_error:
                            logger.error(
                                f"Error during extraction without OCR: {str(extraction_error)}")
                            update_image_status(image_id, "failed")

                except Exception as e:
                    logger.error(
                        f"Error processing image {image_id}: {str(e)}")
                    # エラー情報も保存
                    error_result = {
                        "error": str(e),
                        "words": []
                    }
                    try:
                        update_image_status(image_id, "failed")
                        # OCRが有効な場合のみOCR結果を更新
                        if ENABLE_OCR:
                            update_ocr_result(image_id, error_result, "failed")
                    except Exception as db_error:
                        logger.error(
                            f"Error updating database with failure status: {str(db_error)}")

            # バッチ間に待機時間を入れる（最後のバッチ以外）
            if i + batch_size < len(images):
                wait_time = 3  # 3秒待機
                logger.info(
                    f"Waiting {wait_time} seconds before processing next batch")
                time.sleep(wait_time)

        # ジョブのステータスを更新
        logger.info(f"Completing job {job_id}")
        update_job_status(job_id, "completed")
        logger.info(f"Job {job_id} processing completed")

    except Exception as e:
        logger.error(f"Error in OCR job processing: {str(e)}")
        logger.error(f"Job processing error type: {type(e).__name__}")
        import traceback
        logger.error(f"Job processing traceback: {traceback.format_exc()}")
        try:
            update_job_status(job_id, "failed")
        except Exception as update_error:
            logger.error(
                f"Error updating job status to failed: {str(update_error)}")


@app.get("/ocr/status/{job_id}")
async def get_ocr_status(job_id: str):
    try:
        # ジョブの状態を取得
        job = get_job(job_id)
        job_status = job.get("status")

        # 関連する画像の状態を取得
        images = get_images_by_job_id(job_id)

        return {"status": job_status, "images": images}
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/ocr/result/{image_id}")
async def get_ocr_result(image_id: str):
    try:
        image_data = get_image(image_id)

        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")

        ocr_result = image_data.get("ocr_result")

        # OCR無効時はocr_resultが存在しない
        if ocr_result is None:
            ocr_result = {}

        # 画像URLをレスポンスに追加
        image_url = f"{os.getenv('API_BASE_URL', '')}/image/{image_id}"

        return {
            "filename": image_data.get("filename"),
            "s3_key": image_data.get("s3_key"),
            "uploadTime": image_data.get("upload_time"),
            "status": image_data.get("status"),
            "ocrResult": ocr_result,
            "imageUrl": image_url
        }
    except Exception as e:
        logger.error(f"Error getting OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/image/{image_id}")
async def get_image_endpoint(image_id: str):
    try:
        # データベースからS3キーを取得
        image_data = get_image(image_id)

        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")

        s3_key = image_data.get("s3_key")
        converted_s3_key = image_data.get("converted_s3_key")

        # 変換後の画像がある場合は、常に環境変数のバケットから取得
        if converted_s3_key:
            bucket_name = os.getenv("BUCKET_NAME")
            s3_key = converted_s3_key
            logger.info(f"変換後の画像を取得します: {bucket_name}/{s3_key}")
        else:
            # 変換前の元画像の場合は、アプリ設定のバケットから取得
            app_name = image_data.get("app_name", DEFAULT_APP)
            input_methods = get_app_input_methods(app_name)

            # S3 URIからバケット名を取得
            bucket_name = os.getenv("BUCKET_NAME")  # デフォルトバケット
            if input_methods.get("s3_sync", False) and input_methods.get("s3_uri"):
                s3_uri = input_methods["s3_uri"]
                if s3_uri.startswith("s3://"):
                    parts = s3_uri[5:].split('/', 1)
                    if len(parts) > 0:
                        bucket_name = parts[0]

            logger.info(f"元画像を取得します: {bucket_name}/{s3_key}")

        # S3から画像データを取得
        s3_response = s3_client.get_object(
            Bucket=bucket_name,
            Key=s3_key
        )

        # 画像データを返す
        content_type = s3_response['ContentType']
        image_data = s3_response['Body'].read()

        return StreamingResponse(BytesIO(image_data), media_type=content_type)

    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving image: {str(e)}")


@app.post("/ocr/edit/{image_id}")
async def update_ocr_result_endpoint(image_id: str, edited_ocr_data: dict):
    try:
        # OCR結果を更新
        update_ocr_result(image_id, edited_ocr_data)
        return {"status": "success", "message": "OCR results updated successfully"}
    except Exception as e:
        logger.error(f"Error updating OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/ocr/extract/{image_id}")
async def get_extracted_information(image_id: str):
    try:
        # 画像情報を取得
        image_data = get_image(image_id)

        if not image_data:
            logger.warning(f"画像が見つかりません (image_id: {image_id})")
            raise HTTPException(status_code=404, detail="画像が見つかりません")

        # アプリ名を取得（なければデフォルト）
        app_name = image_data.get("app_name", DEFAULT_APP)

        # アプリの表示名を取得
        app_display_name = get_app_display_name(app_name)

        # このアプリ用の抽出フィールド定義を取得
        app_extraction_fields = get_extraction_fields_for_app(app_name)[
            "fields"]

        # 抽出処理が完了していない場合
        extraction_status = image_data.get("extraction_status")
        if extraction_status != "completed":
            logger.info(f"抽出処理が完了していません (status: {extraction_status})")
            return {
                "extracted_info": {},
                "mapping": {},
                "status": extraction_status or "not_started",
                "app_name": app_name,
                "app_display_name": app_display_name,
                "fields": app_extraction_fields
            }

        extracted_info = image_data.get("extracted_info", {})
        extraction_mapping = image_data.get("extraction_mapping", {})

        logger.info(
            f"DBから取得した抽出情報 (型: {type(extracted_info)}): {extracted_info}")
        logger.info(
            f"DBから取得したマッピング (型: {type(extraction_mapping)}): {extraction_mapping}")

        if not isinstance(extracted_info, dict):
            logger.warning(f"抽出情報が辞書型ではありません (型: {type(extracted_info)})")
            extracted_info = {}

        if not isinstance(extraction_mapping, dict):
            logger.warning(
                f"マッピング情報が辞書型ではありません (型: {type(extraction_mapping)})")
            extraction_mapping = {}

        mapping = extraction_mapping if isinstance(
            extraction_mapping, dict) else {}

        # マッピングが空の場合、デフォルト構造を提供
        if not mapping:
            logger.warning("マッピングが空なので、デフォルト構造を使用します")
            field_names = get_field_names_for_app(app_name)
            mapping = {field_name: [] for field_name in field_names}

        # 最終的なレスポンスデータをログ出力
        response_data = {
            "extracted_info": extracted_info,
            "mapping": mapping,
            "status": "completed",
            "app_name": app_name,
            "app_display_name": app_display_name,
            "fields": app_extraction_fields
        }

        # Decimal型を変換してからJSON化
        try:
            logger.info(
                f"レスポンスデータ: {json.dumps(decimal_to_float(response_data))}")
        except Exception as log_error:
            logger.warning(f"レスポンスデータのログ出力に失敗: {str(log_error)}")

        # レスポンス時もDecimal型を変換
        return decimal_to_float(response_data)

    except Exception as e:
        logger.error(f"抽出情報の取得エラー: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


@app.post("/ocr/extract/{image_id}")
async def extract_information(image_id: str, extraction_data: dict):
    try:
        # フロントエンドから送られてきたOCRデータを使用
        ocr_text = ""
        ocr_words = []
        if "words" in extraction_data:
            # OCRデータから結合テキストを作成
            ocr_words = extraction_data.get("words", [])
            ocr_text = "\n".join([word.get("content", "")
                                 for word in ocr_words])

            # OCRデータを保存（既存のデータを上書きしない場合）
            image_data = get_image(image_id)
            if not image_data or not image_data.get("ocr_result"):
                update_ocr_result(image_id, {"words": ocr_words})

        # テキストが提供されていない場合はDBから取得
        if not ocr_text or not ocr_words:
            image_data = get_image(image_id)

            if not image_data:
                raise HTTPException(status_code=404, detail="Image not found")

            ocr_result = image_data.get("ocr_result")
            if ocr_result and isinstance(ocr_result, dict):
                try:
                    if "words" in ocr_result and ocr_result["words"]:
                        ocr_words = ocr_result["words"]
                        ocr_text = "\n".join(
                            [word.get("content", "") for word in ocr_words])
                except Exception as e:
                    logger.error(
                        f"Error extracting text from OCR result: {str(e)}")

            # 生成されたテキストを優先（OCRテキストがない場合）
            generated_text = image_data.get("generated_text")
            if not ocr_text and generated_text:
                ocr_text = generated_text
                # テキストからダミーのocr_wordsを生成
                ocr_words = []
                for i, line in enumerate(ocr_text.split("\n")):
                    if line.strip():
                        ocr_words.append({"content": line, "id": i})

        if not ocr_text:
            raise HTTPException(
                status_code=400, detail="No text available for extraction")

        # 状態を更新
        update_image_status(image_id, "processing")
        update_ocr_result(image_id, {"words": ocr_words}, "processing")

        # インデックス付きテキストを作成
        indexed_text_lines = []
        for i, word in enumerate(ocr_words):
            if "content" in word and word["content"]:
                indexed_text_lines.append(f"[{i}] {word['content']}")

        indexed_text = "\n".join(indexed_text_lines)

        # インデックス付きテキストが作成できなかった場合でも、空ではなくログを出力
        if not indexed_text:
            logger.warning(
                f"Could not create indexed text for image {image_id}")
            indexed_text = f"[0] {ocr_text}"

        # バックグラウンドタスクとして情報抽出処理を実行
        task_id = background_task.add_task(
            process_information_extraction_with_ocr,
            image_id,
            {"words": ocr_words},
            ocr_text
        )

        return {"status": "success", "message": "Information extraction started", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error starting information extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/ocr/extract/status/{image_id}")
async def get_extraction_status(image_id: str):
    try:
        # 画像情報を取得
        image_data = get_image(image_id)

        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")

        return {"status": image_data.get("extraction_status") or "not_started"}
    except Exception as e:
        logger.error(f"Error getting extraction status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/ocr/extract/edit/{image_id}")
async def update_extracted_info_endpoint(image_id: str, edited_data: dict):
    try:
        # 抽出情報を更新
        extracted_info = edited_data.get("extracted_info", {})
        mapping = edited_data.get("mapping", {})

        update_extracted_info(image_id, extracted_info, mapping)

        return {"status": "success", "message": "抽出情報が正常に更新されました"}
    except Exception as e:
        logger.error(f"抽出情報の更新エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


@app.get("/generate-presigned-download-url/{image_id}")
async def generate_presigned_download_url(image_id: str):
    """
    画像ダウンロード用の署名付きURLを生成する
    """
    try:
        # 画像情報を取得
        image_data = get_image(image_id)

        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")

        # 変換後の画像がある場合はそのパスを使用、なければ元ファイルのパス
        converted_s3_key = image_data.get("converted_s3_key")
        s3_key = image_data.get("s3_key")

        if not s3_key:
            raise HTTPException(status_code=404, detail="Image file not found")

        # 変換後の画像がある場合は、常に環境変数のバケットから取得
        if converted_s3_key:
            bucket_name = os.getenv("BUCKET_NAME")
            s3_key = converted_s3_key
            logger.info(f"変換後の画像のダウンロードURLを生成します: {bucket_name}/{s3_key}")
        else:
            # 変換前の元画像の場合は、アプリ設定のバケットから取得
            app_name = image_data.get("app_name", DEFAULT_APP)
            input_methods = get_app_input_methods(app_name)

            # S3 URIからバケット名を取得
            bucket_name = os.getenv("BUCKET_NAME")  # デフォルトバケット
            if input_methods.get("s3_sync", False) and input_methods.get("s3_uri"):
                s3_uri = input_methods["s3_uri"]
                if s3_uri.startswith("s3://"):
                    parts = s3_uri[5:].split('/', 1)
                    if len(parts) > 0:
                        bucket_name = parts[0]

            logger.info(f"元画像のダウンロードURLを生成します: {bucket_name}/{s3_key}")

        # S3オブジェクトのContent-Typeを取得
        try:
            s3_response = s3_client.head_object(
                Bucket=bucket_name,
                Key=s3_key
            )
            content_type = s3_response.get(
                'ContentType', 'application/octet-stream')
        except ClientError:
            content_type = 'application/octet-stream'

        # 署名付きURLの生成（有効期限は1時間）
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': s3_key,
                'ResponseContentType': content_type,
                'ResponseCacheControl': 'no-cache'  # キャッシュ制御を追加
            },
            ExpiresIn=3600,  # 1時間
            HttpMethod='GET'
        )

        return {
            "presigned_url": presigned_url,
            "content_type": content_type,
            "filename": image_data.get("filename"),
            "is_converted": "converted_s3_key" in image_data
        }

    except Exception as e:
        logger.error(f"Error generating presigned download URL: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error: {str(e)}")


@app.post("/apps")
async def create_app(app_data: dict):
    """新しいアプリを作成または更新する"""
    try:
        app_name = app_data.get("name")
        if not app_name:
            raise HTTPException(status_code=400, detail="アプリ名が指定されていません")

        # 必須フィールドの検証
        required_fields = ["display_name", "fields"]
        for field in required_fields:
            if field not in app_data:
                raise HTTPException(
                    status_code=400, detail=f"必須フィールドがありません: {field}")

        # アプリスキーマを更新
        success = update_app_schema(app_name, app_data)

        if success:
            return {"status": "success", "message": f"アプリ '{app_name}' を作成/更新しました"}
        else:
            raise HTTPException(status_code=500, detail="アプリの作成/更新に失敗しました")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"アプリ作成/更新エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


@app.delete("/apps/{app_name}")
async def delete_app(app_name: str):
    """アプリを削除する"""
    try:
        # デフォルトアプリは削除不可
        if app_name == DEFAULT_APP:
            raise HTTPException(
                status_code=400, detail=f"デフォルトアプリ '{DEFAULT_APP}' は削除できません")

        # 関連データを削除
        logger.info(f"アプリ '{app_name}' の削除を開始します")

        # 1. 関連する画像データを削除
        images_deleted = delete_images_by_app_name(app_name)
        if not images_deleted:
            logger.warning(f"画像データの削除に失敗しました (app_name: {app_name})")

        # 2. 関連するジョブデータを削除
        jobs_deleted = delete_jobs_by_app_name(app_name)
        if not jobs_deleted:
            logger.warning(f"ジョブデータの削除に失敗しました (app_name: {app_name})")

        # 3. アプリスキーマを削除
        schema_deleted = delete_app_schema(app_name)
        if not schema_deleted:
            raise HTTPException(status_code=500, detail="アプリスキーマの削除に失敗しました")

        logger.info(f"アプリ '{app_name}' とその関連データを削除しました")
        return {"status": "success", "message": f"アプリ '{app_name}' を削除しました"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"アプリ削除エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


@app.post("/ocr/yomitoku")
async def perform_yomitoku_ocr_endpoint(file: UploadFile = File(...)):
    try:
        # ファイルを読み込む
        image_data = await file.read()

        # OCR処理を呼び出す
        result = perform_ocr(image_data)

        return result
    except Exception as e:
        logger.error(f"OCR処理中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR処理エラー: {str(e)}")


# スキーマ生成・保存用のリクエストモデル
class SchemaGenerateRequest(BaseModel):
    s3_key: str
    filename: str
    instructions: Optional[str] = None


class SchemaSaveRequest(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    fields: List[Dict[str, Any]]
    input_methods: Dict[str, Any]


@app.post("/schema/save")
async def save_schema(request: SchemaSaveRequest):
    """
    スキーマを保存する
    """
    try:
        # 入力バリデーション
        if not request.name or not request.display_name:
            raise HTTPException(status_code=400, detail="アプリ名と表示名は必須です")

        # アプリ名のバリデーション（英数字とアンダースコアのみ）
        if not re.match(r'^[a-zA-Z0-9_]+$', request.name):
            raise HTTPException(
                status_code=400, detail="アプリ名は英数字とアンダースコアのみ使用できます")

        # 入力方法のバリデーション
        if not request.input_methods.get("file_upload", False) and not request.input_methods.get("s3_sync", False):
            raise HTTPException(
                status_code=400, detail="ファイルアップロードまたはS3同期のいずれかを有効にする必要があります")

        # S3同期が有効な場合、S3 URIが必要
        if request.input_methods.get("s3_sync", False) and not request.input_methods.get("s3_uri"):
            raise HTTPException(
                status_code=400, detail="S3同期が有効な場合、S3 URIを指定する必要があります")

        # スキーマデータを作成
        app_data = {
            "name": request.name,
            "display_name": request.display_name,
            "description": request.description or f"{request.display_name}からの情報抽出",
            "fields": request.fields,
            "input_methods": request.input_methods
        }

        # スキーマを保存
        from app_schema import update_app_schema
        success = update_app_schema(request.name, app_data)

        if success:
            return {"status": "success", "message": f"スキーマ '{request.name}' を保存しました"}
        else:
            raise HTTPException(status_code=500, detail="スキーマの保存に失敗しました")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"スキーマ保存エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"スキーマの保存に失敗しました: {str(e)}")


# スキーマ生成用のpresigned URL生成エンドポイント
@app.post("/schema/generate-presigned-url")
async def generate_schema_presigned_url(request: PresignedUrlRequest):
    """スキーマ生成用のファイルアップロード用presigned URLを生成"""
    try:
        # 一意のS3キーを生成
        image_id = str(uuid.uuid4())
        s3_key = f"schema-uploads/{datetime.now().isoformat()}_{request.filename}"

        # 署名付きURLの生成（有効期限は15分）
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': os.getenv("BUCKET_NAME"),
                'Key': s3_key,
                'ContentType': request.content_type,
            },
            ExpiresIn=900,  # 15分
            HttpMethod='PUT'
        )

        return PresignedUrlResponse(
            presigned_url=presigned_url,
            s3_key=s3_key,
            image_id=image_id
        )

    except Exception as e:
        logger.error(f"署名付きURL生成エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"署名付きURLの生成に失敗しました: {str(e)}")


@app.post("/schema/generate")
async def generate_schema(request: SchemaGenerateRequest):
    """S3にアップロードされた画像からスキーマを生成"""
    try:
        # S3からファイルを取得
        try:
            s3_response = s3_client.get_object(
                Bucket=os.getenv("BUCKET_NAME"),
                Key=request.s3_key
            )
            file_data = s3_response['Body'].read()
        except Exception as e:
            logger.error(f"S3からのファイル取得エラー: {str(e)}")
            raise HTTPException(status_code=404, detail="ファイルが見つかりません")

        # ファイルの種類を拡張子で判定
        _, ext = os.path.splitext(request.filename)
        ext = ext.lower()

        # PDFの場合は画像に変換
        if ext == '.pdf':
            try:
                pdf_document = fitz.open(stream=file_data, filetype="pdf")
                if pdf_document.page_count > 0:
                    page = pdf_document[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    file_data = pix.tobytes("jpeg")
                else:
                    raise HTTPException(
                        status_code=400, detail="PDFにページがありません")
            except Exception as e:
                logger.error(f"PDF変換エラー: {str(e)}")
                raise HTTPException(
                    status_code=400, detail="PDFの変換に失敗しました。有効なPDFファイルをアップロードしてください。")
        elif ext not in ['.jpg', '.jpeg', '.png', '.gif']:
            raise HTTPException(
                status_code=400, detail="サポートされていないファイル形式です。JPG、PNG、GIF、PDFのみ対応しています。")

        # スキーマフィールドを生成
        from ocr import generate_schema_fields_from_image
        schema = generate_schema_fields_from_image(
            file_data, request.instructions)

        # 常に {"fields": [...]} の形式で返す
        if "fields" not in schema:
            return {"fields": []}

        return schema

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"スキーマ生成エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"スキーマの生成に失敗しました: {str(e)}")


@app.put("/schema/update/{app_name}")
async def update_schema(app_name: str, request: SchemaSaveRequest):
    """
    既存のスキーマを更新する
    """
    try:
        # 入力バリデーション
        if not request.name or not request.display_name:
            raise HTTPException(status_code=400, detail="アプリ名と表示名は必須です")

        # アプリ名のバリデーション（英数字とアンダースコアのみ）
        if not re.match(r'^[a-zA-Z0-9_]+$', request.name):
            raise HTTPException(
                status_code=400, detail="アプリ名は英数字とアンダースコアのみ使用できます")

        # 入力方法のバリデーション
        if not request.input_methods.get("file_upload", False) and not request.input_methods.get("s3_sync", False):
            raise HTTPException(
                status_code=400, detail="ファイルアップロードまたはS3同期のいずれかを有効にする必要があります")

        # S3同期が有効な場合、S3 URIが必要
        if request.input_methods.get("s3_sync", False) and not request.input_methods.get("s3_uri"):
            raise HTTPException(
                status_code=400, detail="S3同期が有効な場合、S3 URIを指定する必要があります")

        # スキーマデータを作成
        app_data = {
            "name": request.name,
            "display_name": request.display_name,
            "description": request.description or f"{request.display_name}からの情報抽出",
            "fields": request.fields,
            "input_methods": request.input_methods
        }

        # スキーマを更新
        from app_schema import update_app_schema
        success = update_app_schema(app_name, app_data)

        if success:
            return {"status": "success", "message": f"スキーマ '{app_name}' を更新しました"}
        else:
            raise HTTPException(status_code=500, detail="スキーマの更新に失敗しました")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"スキーマ更新エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"スキーマの更新に失敗しました: {str(e)}")
