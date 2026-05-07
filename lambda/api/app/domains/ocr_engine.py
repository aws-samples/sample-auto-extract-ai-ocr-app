"""OCR エンジン — 純粋な OCR レスポンス整形ロジック

SageMaker / S3 への直接アクセスは行わない。
外部サービス呼び出しは ocr_service.py が担当する。
"""
import logging

logger = logging.getLogger(__name__)


def parse_ocr_response(response_body: dict) -> dict:
    """SageMaker OCR レスポンスを整形・軽量化する（純粋関数）

    Args:
        response_body: SageMaker エンドポイントからの生レスポンス

    Returns:
        整形済み OCR 結果 {"text": str, "words": list, "word_count": int}
        エラー時は {"error": str, "text": "", "words": [], "word_count": 0}
    """
    if "error" in response_body:
        logger.error(f"SageMakerエンドポイントからエラーが返されました: {response_body['error']}")
        return response_body

    # OCR結果を軽量化（不要なフィールドを削除）
    if "words" in response_body:
        simplified_words = []
        for word in response_body["words"]:
            simplified_word = {
                "id": word["id"],
                "content": word["content"],
                "points": word["points"],
            }
            if "direction" in word:
                simplified_word["direction"] = word["direction"]
            simplified_words.append(simplified_word)
        response_body["words"] = simplified_words

    words = response_body.get("words", [])
    full_text = " ".join([word.get("content", "") for word in words])

    return {
        "text": full_text,
        "words": words,
        "word_count": len(words),
    }
