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


def safe_get_from_dynamo_data(data, key, default=None):
    """
    DynamoDBのデータを安全に取得

    Args:
        data: DynamoDBから取得したデータ
        key: 取得したいキー
        default: デフォルト値

    Returns:
        取得した値またはデフォルト値
    """
    try:
        if isinstance(data, dict):
            return data.get(key, default)
        else:
            return default
    except Exception as e:
        logger.warning(f"DynamoDBデータ取得エラー: {str(e)}")
        return default


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
