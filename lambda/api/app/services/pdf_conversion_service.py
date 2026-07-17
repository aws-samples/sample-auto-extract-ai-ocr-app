"""PDF 関連サービス

PDF→画像変換のオーケストレーションと、複数ページ PDF の親ドキュメントステータス連動を担当。
S3 からの PDF 取得、ページごとの画像変換、S3 アップロード、DynamoDB 更新を行う。
純粋な画像変換処理は utils/pdf.py の関数を使用する。
"""
import logging
import uuid
import os
from datetime import datetime
import fitz
from PIL import Image
import tempfile
import io

from clients import s3_client
from config import settings
from exceptions import NotFoundError, BadRequestError
from repositories import (
    get_image, update_image_status, update_converted_image,
    update_ocr_result, update_parent_document_status,
    create_individual_page_record, get_app_input_methods,
    get_children_by_parent_id, update_agent_status,
)
from domains.image_status import determine_parent_status, determine_parent_agent_status, ImageStatus, AgentStatus, PageProcessingMode
from utils.helpers import resize_image

logger = logging.getLogger(__name__)


def convert_pdf_to_image(image_id: str, s3_key: str):
    """PDFを画像に変換し、S3にアップロードする（処理モード対応版）"""
    try:
        logger.info(f"PDFの変換を開始します: {image_id}, {s3_key}")

        image_data = get_image(image_id)
        if not image_data:
            raise NotFoundError(f"Image not found: {image_id}")
        app_name = image_data.get("app_name")
        if not app_name:
            raise NotFoundError(f"app_name not found for image {image_id}")

        processing_mode = image_data.get("page_processing_mode", PageProcessingMode.COMBINED)
        input_methods = get_app_input_methods(app_name)

        # S3 URIからバケット名を取得
        bucket_name = settings.BUCKET_NAME
        if input_methods.get("s3_sync", False) and input_methods.get("s3_uri"):
            s3_uri = input_methods["s3_uri"]
            if s3_uri.startswith("s3://"):
                parts = s3_uri[5:].split("/", 1)
                if len(parts) > 0:
                    bucket_name = parts[0]

        # S3からPDFファイルを取得
        s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        file_content = s3_response["Body"].read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_content)
            temp_pdf_path = temp_pdf.name

        try:
            pdf_document = fitz.open(temp_pdf_path)
            if pdf_document.page_count == 0:
                raise BadRequestError("PDF has no pages")

            upload_bucket = settings.BUCKET_NAME
            if not upload_bucket:
                raise RuntimeError("BUCKET_NAME environment variable is not set")

            if processing_mode == PageProcessingMode.COMBINED:
                _process_combined_pages(pdf_document, image_id, s3_key, upload_bucket)
            elif processing_mode == PageProcessingMode.INDIVIDUAL and pdf_document.page_count == 1:
                _process_combined_pages(pdf_document, image_id, s3_key, upload_bucket)
            else:
                _process_individual_pages(pdf_document, image_id, s3_key, upload_bucket)

            pdf_document.close()
        finally:
            try:
                os.unlink(temp_pdf_path)
            except Exception as e:
                logger.warning(f"一時ファイルの削除に失敗しました: {str(e)}")

    except Exception as e:
        logger.error(f"PDF変換エラー: {str(e)}")
        update_image_status(image_id, ImageStatus.FAILED)
        try:
            update_ocr_result(image_id, {"error": str(e), "timestamp": datetime.now().isoformat()})
        except Exception as db_error:
            logger.error(f"エラー情報の保存に失敗しました: {str(db_error)}")


def _convert_page_to_jpeg(pdf_document, page_num: int) -> tuple[bytes, tuple, tuple, bool]:
    """PDF ページを JPEG に変換しリサイズする（内部ヘルパー）

    Returns:
        (image_data, original_size, new_size, was_resized)
    """
    page = pdf_document[page_num]
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG", quality=95)
    img_data = img_byte_arr.getvalue()
    original_size = (pix.width, pix.height)

    try:
        resized_data, was_resized, orig_size, new_size = resize_image(img_data)
    except ImportError:
        resized_data = img_data
        was_resized = False
        orig_size = original_size
        new_size = original_size

    final_data = resized_data if was_resized else img_data
    return final_data, orig_size if was_resized else original_size, new_size if was_resized else original_size, was_resized


def _process_combined_pages(pdf_document, image_id: str, s3_key: str, upload_bucket: str):
    """複数ページPDFを複数画像として処理する"""
    total_pages = pdf_document.page_count
    logger.info(f"複数画像処理を開始: {total_pages}ページ")

    if total_pages > 10:
        raise BadRequestError(f"PDF has too many pages ({total_pages}). Maximum supported: 10")

    if total_pages == 1:
        return _process_single_page_combined(pdf_document, image_id, s3_key, upload_bucket)

    page_s3_keys = []
    filename_base = os.path.splitext(os.path.basename(s3_key))[0]

    for page_num in range(total_pages):
        img_data, orig_size, new_size, was_resized = _convert_page_to_jpeg(pdf_document, page_num)
        page_s3_key = f"converted/{datetime.now().isoformat()}_{filename_base}_page_{page_num + 1}.jpeg"
        s3_client.put_object(Bucket=upload_bucket, Key=page_s3_key, Body=img_data, ContentType="image/jpeg")
        page_s3_keys.append(page_s3_key)
        logger.info(f"ページ {page_num + 1}/{total_pages} 保存完了: {page_s3_key}")

    update_converted_image(image_id, page_s3_keys, ImageStatus.PENDING, None, None, page_processing_mode=PageProcessingMode.COMBINED, total_pages=total_pages)
    logger.info(f"複数画像処理完了: {image_id}, {total_pages}ページ")


def _process_single_page_combined(pdf_document, image_id: str, s3_key: str, upload_bucket: str):
    """単一ページPDFを処理する（統合処理モード）"""
    logger.info("単一ページPDFを処理します（統合モード）")
    img_data, orig_size, new_size, was_resized = _convert_page_to_jpeg(pdf_document, 0)
    filename_base = os.path.splitext(os.path.basename(s3_key))[0]
    converted_s3_key = f"converted/{datetime.now().isoformat()}_{filename_base}_single.jpeg"
    s3_client.put_object(Bucket=upload_bucket, Key=converted_s3_key, Body=img_data, ContentType="image/jpeg")
    update_converted_image(image_id, [converted_s3_key], ImageStatus.PENDING, orig_size, new_size, page_processing_mode=PageProcessingMode.COMBINED, total_pages=1)
    logger.info(f"単一ページ処理完了: {image_id}")


def _process_individual_pages(pdf_document, parent_image_id: str, s3_key: str, upload_bucket: str):
    """複数ページPDFを個別ページとして処理する"""
    total_pages = pdf_document.page_count
    logger.info(f"個別処理を開始: {total_pages}ページ")
    update_parent_document_status(parent_image_id, ImageStatus.CONVERTING, total_pages=total_pages)

    created_page_ids = []
    for page_num in range(total_pages):
        try:
            page_id = _create_individual_page(pdf_document, page_num, parent_image_id, s3_key, upload_bucket, total_pages)
            created_page_ids.append(page_id)
            logger.info(f"個別ページ {page_num + 1}/{total_pages} 作成完了: {page_id}")
        except Exception as page_error:
            logger.error(f"ページ {page_num + 1} の処理でエラー: {str(page_error)}")
            continue

    if created_page_ids:
        update_parent_document_status(parent_image_id, ImageStatus.PENDING)
    else:
        update_parent_document_status(parent_image_id, ImageStatus.FAILED)
        logger.error("個別処理失敗: ページが作成されませんでした")


def _create_individual_page(pdf_document, page_num: int, parent_image_id: str,
                            s3_key: str, upload_bucket: str, total_pages: int) -> str:
    """個別ページを作成・保存する"""
    img_data, orig_size, new_size, was_resized = _convert_page_to_jpeg(pdf_document, page_num)
    filename_base = os.path.splitext(os.path.basename(s3_key))[0]
    page_s3_key = f"converted/{datetime.now().isoformat()}_{filename_base}_page_{page_num + 1}.jpeg"
    s3_client.put_object(Bucket=upload_bucket, Key=page_s3_key, Body=img_data, ContentType="image/jpeg")

    page_id = str(uuid.uuid4())
    parent_data = get_image(parent_image_id)
    create_individual_page_record(
        page_id=page_id, parent_image_id=parent_image_id,
        filename=parent_data.get("filename"), converted_s3_key=page_s3_key,
        page_number=page_num + 1, total_pages=total_pages,
        app_name=parent_data.get("app_name"),
        original_size=orig_size, new_size=new_size,
        uploaded_by=parent_data.get("uploaded_by"),
    )
    return page_id


def sync_parent_status(image_id: str) -> None:
    """子ページのステータス変更後に親ドキュメントのステータスを連動更新する

    複数ページ PDF の子ページ（個別ページ画像）のステータスが変わった際に、
    親ドキュメント（PDF 全体）のステータスを子ページの状態から再計算して更新する。
    OcrService / ExtractionService の両方から呼ばれる共通処理。
    """
    try:
        image_data = get_image(image_id)
        if not image_data or not image_data.get("parent_document_id"):
            return

        parent_id = image_data["parent_document_id"]
        children = get_children_by_parent_id(parent_id)
        new_status = determine_parent_status(children)

        parent_data = get_image(parent_id)
        if parent_data and parent_data.get("status") != new_status:
            update_parent_document_status(parent_id, new_status)
            logger.info(f"親ドキュメントステータス更新: {parent_id} -> {new_status}")

    except Exception as e:
        logger.error(f"親ステータス更新エラー: {str(e)}")


def sync_parent_agent_status(image_id: str) -> None:
    """子ページの agent_status 変更後に親ドキュメントの agent_status を連動更新する"""
    try:
        image_data = get_image(image_id)
        if not image_data or not image_data.get("parent_document_id"):
            return

        parent_id = image_data["parent_document_id"]
        children = get_children_by_parent_id(parent_id)
        new_agent_status = determine_parent_agent_status(children)

        parent_data = get_image(parent_id)
        current_agent_status = parent_data.get("agent_status") or AgentStatus.IDLE if parent_data else AgentStatus.IDLE
        if current_agent_status != new_agent_status:
            update_agent_status(parent_id, new_agent_status)
            logger.info(f"親ドキュメント agent_status 更新: {parent_id} -> {new_agent_status}")

    except Exception as e:
        logger.error(f"親 agent_status 更新エラー: {str(e)}")
