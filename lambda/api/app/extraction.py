import json
import logging
import os
import boto3
from decimal import Decimal

from app_schema import get_app_schema
from database import get_image, update_extracted_info, update_image_status

# ログ設定
logger = logging.getLogger(__name__)

# Bedrock クライアント
bedrock_client = boto3.client('bedrock-runtime')
s3_client = boto3.client('s3')


def extract_info_with_single_image(image_id: str, ocr_result: dict, ocr_text: str):
    """
    単一画像+OCR結果での情報抽出
    """
    try:
        logger.info(f"単一画像情報抽出を開始: {image_id}")
        
        # 画像データを取得
        image_data = get_image(image_id)
        app_name = image_data.get("app_name", "default")
        
        # アプリケーションスキーマを取得
        app_schema = get_app_schema(app_name)
        if not app_schema:
            logger.error(f"App schema not found: {app_name}")
            return
        
        # 従来の処理ロジックを使用
        from ocr import process_information_extraction_with_ocr
        process_information_extraction_with_ocr(image_id, ocr_result, ocr_text)
        
        logger.info(f"単一画像情報抽出完了: {image_id}")
        
    except Exception as e:
        logger.error(f"単一画像情報抽出エラー: {str(e)}")
        update_image_status(image_id, "failed")
        raise


def extract_info_with_multiimage(image_id: str):
    """
    複数画像+OCR結果での情報抽出
    """
    try:
        logger.info(f"複数画像情報抽出を開始: {image_id}")
        
        # 画像データを取得
        image_data = get_image(image_id)
        app_name = image_data.get("app_name", "default")
        
        # アプリケーションスキーマを取得
        app_schema = get_app_schema(app_name)
        if not app_schema:
            logger.error(f"App schema not found: {app_name}")
            return
        
        # 抽出指示を作成
        instructions = f"以下のスキーマに従って、文書から情報を抽出してください。"
        
        # 複数画像情報抽出を実行
        extracted_text = extract_info_multiimage_core(
            image_id, 
            app_schema.get("fields", {}), 
            instructions
        )
        
        # JSON形式の結果を解析
        try:
            extracted_info = json.loads(extracted_text)
        except json.JSONDecodeError:
            # JSON解析に失敗した場合は、テキストから情報を抽出
            logger.warning("JSON解析に失敗、テキストから情報を抽出します")
            extracted_info = {"raw_response": extracted_text}
        
        # 結果を保存
        update_extracted_info(image_id, extracted_info, {}, "completed")
        
        logger.info(f"複数画像情報抽出完了: {image_id}")
        
    except Exception as e:
        logger.error(f"複数画像情報抽出エラー: {str(e)}")
        update_image_status(image_id, "failed")
        raise


def extract_info_without_ocr(image_id: str):
    """
    OCRなしでの情報抽出（画像のみ → LLM）
    """
    try:
        logger.info(f"OCRなし情報抽出を開始: {image_id}")
        
        # 従来の処理ロジックを使用
        from ocr import process_information_extraction_without_ocr
        process_information_extraction_without_ocr(image_id)
        
        logger.info(f"OCRなし情報抽出完了: {image_id}")
        
    except Exception as e:
        logger.error(f"OCRなし情報抽出エラー: {str(e)}")
        update_image_status(image_id, "failed")
        raise


def extract_info_multiimage_core(image_id: str, schema: dict, instructions: str):
    """
    複数画像での情報抽出のコア処理
    """
    try:
        logger.info(f"複数画像情報抽出コア処理を開始: {image_id}")
        
        # 画像データとOCR結果を取得
        image_data = get_image(image_id)
        converted_s3_keys = image_data.get("converted_s3_key")
        
        if not converted_s3_keys:
            raise ValueError("変換済み画像が見つかりません")
        
        # リスト形式でない場合は単一画像として扱う
        if not isinstance(converted_s3_keys, list):
            converted_s3_keys = [converted_s3_keys]
        
        # OCR結果を取得
        ocr_results = get_multipage_ocr_results(image_id)
        
        # プロンプト生成
        prompt = create_multiimage_extraction_prompt(ocr_results, schema, instructions)
        
        # 複数画像を取得
        page_images = []
        for s3_key in converted_s3_keys:
            image_bytes = get_s3_object_bytes(s3_key)
            page_images.append(image_bytes)
        
        logger.info(f"画像数: {len(page_images)}, OCRページ数: {len(ocr_results)}")
        
        # Bedrock APIに送信
        content = [{"text": prompt}]
        
        # 各ページの画像を追加
        for i, image_bytes in enumerate(page_images):
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": image_bytes}
                }
            })
        
        # Bedrock呼び出し
        response = bedrock_client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 4000, "temperature": 0.1}
        )
        
        # レスポンスを解析
        response_text = response["output"]["message"]["content"][0]["text"]
        logger.info(f"複数画像情報抽出コア処理完了: {image_id}")
        
        return response_text
        
    except Exception as e:
        logger.error(f"複数画像情報抽出コア処理エラー: {str(e)}")
        raise


def get_multipage_ocr_results(image_id: str) -> list:
    """複数ページOCR結果を取得"""
    try:
        image_data = get_image(image_id)
        ocr_result = image_data.get("ocr_result", {})
        
        # 複数ページOCR結果を取得
        pages_results = ocr_result.get("pages", [])
        if pages_results:
            return pages_results
        
        # 従来形式の場合は単一ページとして扱う
        words = ocr_result.get("words", [])
        return [{"page": 1, "words": words}]
        
    except Exception as e:
        logger.error(f"複数ページOCR結果取得エラー: {str(e)}")
        return []


def create_multiimage_extraction_prompt(ocr_results: list, schema: dict, instructions: str):
    """複数画像・複数OCR結果用のプロンプト生成"""
    
    # OCR結果をページ別に整理
    ocr_text_by_page = []
    for page_result in ocr_results:
        page_num = page_result.get("page", 1)
        page_words = page_result.get("words", [])
        page_text = extract_text_from_ocr(page_words)
        ocr_text_by_page.append(f"=== ページ {page_num} OCRテキスト ===\n{page_text}")
    
    combined_ocr_text = "\n\n".join(ocr_text_by_page)
    
    prompt = f"""以下は{len(ocr_results)}ページの文書です。各ページの画像とOCRテキストを確認して、指定されたスキーマに従って情報を抽出してください。

{combined_ocr_text}

抽出スキーマ:
{json.dumps(schema, ensure_ascii=False, indent=2)}

抽出指示:
{instructions}

重要な注意事項:
- 複数ページにまたがる情報は統合して抽出してください
- ページ番号や参照情報がある場合は保持してください
- 不明な項目は null または空文字列で返してください
- 必ずJSON形式で回答してください

抽出結果:
"""
    
    return prompt


def extract_text_from_ocr(words: list) -> str:
    """OCR結果から読みやすいテキストを抽出"""
    if not words:
        return "（OCRテキストなし）"
    
    # 座標順にソート（上から下、左から右）
    sorted_words = sorted(words, key=lambda w: (w.get("top", 0), w.get("left", 0)))
    
    # テキストを結合
    text_lines = []
    current_line = []
    current_top = None
    
    for word in sorted_words:
        word_top = word.get("top", 0)
        word_text = word.get("content", "").strip()
        
        if not word_text:
            continue
            
        # 新しい行の判定（Y座標の差が一定以上）
        if current_top is None or abs(word_top - current_top) > 10:
            if current_line:
                text_lines.append(" ".join(current_line))
            current_line = [word_text]
            current_top = word_top
        else:
            current_line.append(word_text)
    
    # 最後の行を追加
    if current_line:
        text_lines.append(" ".join(current_line))
    
    return "\n".join(text_lines)


def get_s3_object_bytes(s3_key: str) -> bytes:
    """S3から画像バイトデータを取得"""
    try:
        bucket_name = os.getenv("BUCKET_NAME")
        s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        return s3_response['Body'].read()
    except Exception as e:
        logger.error(f"S3オブジェクト取得エラー: {s3_key}, {str(e)}")
        raise
