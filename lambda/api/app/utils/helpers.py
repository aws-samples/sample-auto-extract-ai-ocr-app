"""
共通ヘルパー関数
"""
import logging
from decimal import Decimal
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


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


def float_to_decimal(obj):
    """float型をDecimal型に変換してDynamoDB保存可能にする"""
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [float_to_decimal(item) for item in obj]
    elif isinstance(obj, float):
        return Decimal(str(obj))
    else:
        return obj


def resize_image(image_data, max_dimension=1568, min_dimension=200):
    """
    画像をリサイズする関数
    - 長辺が max_dimension を超える場合はリサイズ
    - 短辺が min_dimension より小さい場合は警告
    - アスペクト比は維持
    """
    try:
        img = Image.open(BytesIO(image_data))
        width, height = img.size
        
        # 画像サイズのログ記録
        logger.info(f"元の画像サイズ: {width}x{height}px")
        
        # 画像が小さすぎる場合は警告
        if width < min_dimension or height < min_dimension:
            logger.warning(f"画像サイズが小さすぎます: {width}x{height}px")
        
        # リサイズが必要かチェック
        if width <= max_dimension and height <= max_dimension:
            logger.info("リサイズ不要: 画像サイズは既に最適です")
            return image_data, False, (width, height), (width, height)
        
        # アスペクト比を維持してリサイズ
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        # リサイズ実行
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        logger.info(f"リサイズ後の画像サイズ: {new_width}x{new_height}px")
        
        # BytesIOに保存して返す
        output = BytesIO()
        img_format = img.format or 'JPEG'
        resized_img.save(output, format=img_format)
        output.seek(0)
        
        return output.getvalue(), True, (width, height), (new_width, new_height)
    
    except Exception as e:
        logger.error(f"画像リサイズエラー: {str(e)}")
        # エラーの場合は元の画像を返す
        return image_data, False, None, None


MAX_PAYLOAD_IMAGE_BYTES = 4 * 1024 * 1024  # 4MB (base64で~5.3MB、6MB制限内に収まる)


def compress_image_for_payload(image_data: bytes, max_bytes: int = MAX_PAYLOAD_IMAGE_BYTES) -> bytes:
    """SageMaker invoke_endpoint の 6MB ペイロード制限に収まるよう画像を圧縮する"""
    if len(image_data) <= max_bytes:
        return image_data

    logger.info(f"画像サイズ超過: {len(image_data)} bytes → {max_bytes} bytes 以下に圧縮")

    try:
        img = Image.open(BytesIO(image_data))
        img_format = img.format or 'JPEG'

        if img_format != 'JPEG':
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img_format = 'JPEG'

        for quality in [85, 75, 60, 45, 30]:
            output = BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            if output.tell() <= max_bytes:
                logger.info(f"圧縮完了: quality={quality}, size={output.tell()} bytes")
                return output.getvalue()

        width, height = img.size
        for scale in [0.75, 0.5, 0.35]:
            new_size = (int(width * scale), int(height * scale))
            resized = img.resize(new_size, Image.LANCZOS)
            output = BytesIO()
            resized.save(output, format='JPEG', quality=45, optimize=True)
            if output.tell() <= max_bytes:
                logger.info(f"圧縮完了: scale={scale}, size={output.tell()} bytes")
                return output.getvalue()

        logger.warning(f"最大圧縮でも目標サイズ未達: {output.tell()} bytes")
        return output.getvalue()

    except Exception as e:
        logger.error(f"画像圧縮エラー: {str(e)}")
        return image_data
