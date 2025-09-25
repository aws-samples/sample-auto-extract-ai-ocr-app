"""
PDF処理関連のユーティリティ関数
"""
from clients import s3_client
import logging
import uuid
import os
from datetime import datetime
import boto3
import fitz
from PIL import Image
import tempfile

from config import settings
from database import get_image, update_image_status, update_converted_image, update_ocr_result, update_parent_document_status, create_individual_page_record
from app_schema import DEFAULT_APP, get_app_input_methods
from utils.helpers import resize_image

logger = logging.getLogger(__name__)


def convert_pdf_to_image(image_id: str, s3_key: str):
    """
    PDFを画像に変換し、S3にアップロードする（処理モード対応版）

    Args:
        image_id (str): 画像ID
        s3_key (str): PDFファイルのS3キー
    """
    try:
        logger.info(f"PDFの変換を開始します: {image_id}, {s3_key}")

        # 画像情報を取得してバケット名と処理モードを決定
        image_data = get_image(image_id)
        app_name = image_data.get("app_name", DEFAULT_APP)
        processing_mode = image_data.get("page_processing_mode", "combined")
        input_methods = get_app_input_methods(app_name)

        logger.info(f"処理モード: {processing_mode}")

        # S3 URIからバケット名を取得
        bucket_name = settings.BUCKET_NAME  # デフォルトバケット
        if input_methods.get("s3_sync", False) and input_methods.get("s3_uri"):
            s3_uri = input_methods["s3_uri"]
            if s3_uri.startswith("s3://"):
                parts = s3_uri[5:].split('/', 1)
                if len(parts) > 0:
                    bucket_name = parts[0]

        logger.info(f"S3バケット名: {bucket_name}")

        # S3からPDFファイルを取得
        s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        file_content = s3_response['Body'].read()

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(file_content)
            temp_pdf_path = temp_pdf.name

        try:
            # PDFを開く
            pdf_document = fitz.open(temp_pdf_path)

            if pdf_document.page_count == 0:
                raise ValueError("PDF has no pages")

            # 変換後のファイルは常に環境変数で指定されたバケットに保存
            upload_bucket = settings.BUCKET_NAME
            if not upload_bucket:
                raise ValueError("BUCKET_NAME environment variable is not set")

            logger.info(f"変換後のファイルの保存先バケット: {upload_bucket}")

            # 処理モードに応じて分岐
            if processing_mode == "combined":
                process_combined_pages(
                    pdf_document, image_id, s3_key, upload_bucket)
            elif processing_mode == "individual" and pdf_document.page_count == 1:
                # 1ページの個別処理は統合処理として扱う
                logger.info("1ページの個別処理を統合処理として実行")
                process_combined_pages(
                    pdf_document, image_id, s3_key, upload_bucket)
            else:
                # 2ページ以上の個別処理
                process_individual_pages(
                    pdf_document, image_id, s3_key, upload_bucket)

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
        try:
            error_result = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            update_ocr_result(image_id, error_result, "failed")
        except Exception as db_error:
            logger.error(f"エラー情報の保存に失敗しました: {str(db_error)}")


def process_combined_pages(pdf_document, image_id: str, s3_key: str, upload_bucket: str):
    """
    複数ページPDFを複数画像として処理する（元の実装）
    """
    try:
        total_pages = pdf_document.page_count
        logger.info(f"複数画像処理を開始: {total_pages}ページ")

        # 10ページ制限チェック
        if total_pages > 10:
            raise ValueError(
                f"PDF has too many pages ({total_pages}). Maximum supported: 10")

        if total_pages == 1:
            # 単一ページの場合
            return process_single_page_combined(pdf_document, image_id, s3_key, upload_bucket)

        # 複数ページを個別画像として保存
        page_s3_keys = []
        filename_base = os.path.splitext(os.path.basename(s3_key))[0]

        for page_num in range(total_pages):
            page = pdf_document[page_num]
            pix = page.get_pixmap(dpi=300)  # 高解像度で画像化

            # PILイメージに変換
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # 画像をバイトストリームに変換
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_data = img_byte_arr.getvalue()

            # 画像をリサイズ
            try:
                resized_image_data, was_resized, orig_size, new_size = resize_image(
                    img_data)
            except ImportError:
                # resize_image関数がない場合はそのまま使用
                resized_image_data = img_data
                was_resized = False

            # S3キーを生成
            page_s3_key = f"converted/{datetime.now().isoformat()}_{filename_base}_page_{page_num + 1}.jpeg"

            # S3にアップロード
            s3_client.put_object(
                Bucket=upload_bucket,
                Key=page_s3_key,
                Body=resized_image_data if was_resized else img_data,
                ContentType='image/jpeg'
            )

            page_s3_keys.append(page_s3_key)
            logger.info(
                f"ページ {page_num + 1}/{total_pages} 保存完了: {page_s3_key}")

        # DynamoDBを更新（複数S3キーを保存）
        update_converted_image(
            image_id,
            page_s3_keys,  # リストで保存
            "pending",     # ステータスを pending に更新
            None,  # original_size（複数画像の場合は個別管理）
            None,  # new_size（複数画像の場合は個別管理）
            page_processing_mode="combined",
            total_pages=total_pages
        )
        logger.info(f"複数画像処理完了: {image_id}, {total_pages}ページ")

    except Exception as e:
        logger.error(f"複数画像処理エラー: {str(e)}")
        raise


def process_single_page_combined(pdf_document, image_id: str, s3_key: str, upload_bucket: str):
    """
    単一ページPDFを処理する（統合処理モード）
    """
    try:
        logger.info("単一ページPDFを処理します（統合モード）")

        # ページを画像として処理
        page = pdf_document[0]
        pix = page.get_pixmap(dpi=300)  # 高解像度で画像化

        # バイトデータをBytesIOオブジェクトに変換
        import io
        img_byte_arr = io.BytesIO()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(img_byte_arr, format='JPEG', quality=95)
        img_data = img_byte_arr.getvalue()

        # 元のサイズを記録
        original_size = (pix.width, pix.height)

        # 画像をリサイズ
        try:
            resized_image_data, was_resized, orig_size, new_size = resize_image(
                img_data)
        except ImportError:
            # resize_image関数がない場合はそのまま使用
            resized_image_data = img_data
            was_resized = False
            orig_size = original_size
            new_size = original_size

        # 変換後のS3キーを生成
        filename_base = os.path.splitext(os.path.basename(s3_key))[0]
        converted_s3_key = f"converted/{datetime.now().isoformat()}_{filename_base}_single.jpeg"

        # S3にアップロード（常に環境変数のバケットを使用）
        s3_client.put_object(
            Bucket=upload_bucket,
            Key=converted_s3_key,
            Body=resized_image_data if was_resized else img_data,
            ContentType='image/jpeg'
        )

        # DynamoDBを更新（単一ページでもリスト形式で保存）
        update_converted_image(
            image_id,
            [converted_s3_key],  # 単一ページでもリスト形式
            "pending",
            orig_size if was_resized else original_size,
            new_size if was_resized else original_size,
            page_processing_mode="combined",
            total_pages=1
        )
        logger.info(f"単一ページ処理完了: {image_id}")

    except Exception as e:
        logger.error(f"単一ページ処理エラー: {str(e)}")
        raise


def process_individual_pages(pdf_document, parent_image_id: str, s3_key: str, upload_bucket: str):
    """
    複数ページPDFを個別ページとして処理する
    """
    try:
        total_pages = pdf_document.page_count
        logger.info(f"個別処理を開始: {total_pages}ページ")

        # 親ドキュメントの情報を更新
        update_parent_document_status(
            parent_image_id,
            "converting",
            total_pages=total_pages
        )

        created_page_ids = []

        # 各ページを個別に処理
        for page_num in range(total_pages):
            try:
                page_id = create_individual_page(
                    pdf_document,
                    page_num,
                    parent_image_id,
                    s3_key,
                    upload_bucket,
                    total_pages
                )
                created_page_ids.append(page_id)
                logger.info(
                    f"個別ページ {page_num + 1}/{total_pages} 作成完了: {page_id}")

            except Exception as page_error:
                logger.error(f"ページ {page_num + 1} の処理でエラー: {str(page_error)}")
                # 個別ページのエラーでも処理を続行
                continue

        # 親ドキュメントのステータスを更新
        if created_page_ids:
            update_parent_document_status(parent_image_id, "pending")
            logger.info(f"個別処理完了: {len(created_page_ids)}ページ作成")
        else:
            update_parent_document_status(parent_image_id, "failed")
            logger.error("個別処理失敗: ページが作成されませんでした")

    except Exception as e:
        logger.error(f"個別処理エラー: {str(e)}")
        update_parent_document_status(parent_image_id, "failed")
        raise


def create_individual_page(pdf_document, page_num: int, parent_image_id: str,
                           s3_key: str, upload_bucket: str, total_pages: int):
    """
    個別ページを作成・保存する

    Returns:
        str: 作成されたページのID
    """
    # ページを画像として処理
    page = pdf_document[page_num]
    pix = page.get_pixmap(dpi=300)

    # PILイメージに変換
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 画像をバイトストリームに変換
    import io
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=95)
    img_data = img_byte_arr.getvalue()

    # 元のサイズを記録
    original_size = (pix.width, pix.height)

    # 画像をリサイズ
    try:
        resized_image_data, was_resized, orig_size, new_size = resize_image(
            img_data)
    except ImportError:
        resized_image_data = img_data
        was_resized = False
        orig_size = original_size
        new_size = original_size

    # S3キーを生成
    filename_base = os.path.splitext(os.path.basename(s3_key))[0]
    page_s3_key = f"converted/{datetime.now().isoformat()}_{filename_base}_page_{page_num + 1}.jpeg"

    # S3にアップロード
    s3_client.put_object(
        Bucket=upload_bucket,
        Key=page_s3_key,
        Body=resized_image_data if was_resized else img_data,
        ContentType='image/jpeg'
    )

    # 個別ページレコードを作成
    page_id = str(uuid.uuid4())
    parent_data = get_image(parent_image_id)

    create_individual_page_record(
        page_id=page_id,
        parent_image_id=parent_image_id,
        filename=parent_data.get("filename"),
        converted_s3_key=page_s3_key,
        page_number=page_num + 1,
        total_pages=total_pages,
        app_name=parent_data.get("app_name"),
        original_size=orig_size if was_resized else original_size,
        new_size=new_size if was_resized else original_size
    )

    return page_id
