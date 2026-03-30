"""
PDF処理関連の純粋ユーティリティ関数

副作用のあるオーケストレーション（S3 アップロード、DynamoDB 更新）は
services/pdf_conversion_service.py に移動済み。
"""
import logging
import fitz

logger = logging.getLogger(__name__)


def pdf_page_to_jpeg(pdf_bytes: bytes, page_num: int = 0, dpi: int = 300) -> bytes:
    """PDF の指定ページを JPEG バイトデータに変換する（純粋関数）

    Args:
        pdf_bytes: PDF ファイルのバイトデータ
        page_num: ページ番号（0-indexed）
        dpi: 解像度

    Returns:
        JPEG バイトデータ
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        doc.close()
        raise ValueError("PDF にページがありません")
    if page_num >= doc.page_count:
        doc.close()
        raise ValueError(f"ページ {page_num} は存在しません（全 {doc.page_count} ページ）")
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    jpeg_bytes = pix.tobytes("jpeg")
    doc.close()
    return jpeg_bytes
