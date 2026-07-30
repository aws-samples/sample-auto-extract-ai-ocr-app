"""OCR エンジン — 純粋な OCR レスポンス整形ロジック

SageMaker / S3 への直接アクセスは行わない。
外部サービス呼び出しは ocr_service.py が担当する。
"""
import logging

logger = logging.getLogger(__name__)


def _normalize_points(points):
    """OCR 座標 (points) をピクセル整数に正規化する。

    `[[x, y], ...]` の各座標を int に丸める。数値でない要素や list-of-list でない形は
    そのまま返す。

    Args:
        points: 座標リスト。

    Returns:
        各座標を int に丸めたリスト。正規化できない形はそのまま返す。
    """
    if not isinstance(points, list):
        return points
    normalized = []
    for point in points:
        if isinstance(point, list):
            normalized.append([
                int(round(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for v in point
            ])
        else:
            normalized.append(point)
    return normalized


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
                "points": _normalize_points(word["points"]),
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


def parse_yomitoku_mp_response(response_body: dict) -> dict:
    """Yomitoku Marketplace レスポンスを既存 OCR 結果形式に変換する"""
    if "error" in response_body:
        logger.error(f"Yomitoku MP エラー: {response_body['error']}")
        return {"error": response_body["error"], "text": "", "words": [], "word_count": 0}

    results = response_body.get("result", [])
    if not results:
        logger.error("Yomitoku MP: レスポンスに result フィールドがありません")
        return {"error": "No result from Yomitoku", "text": "", "words": [], "word_count": 0}

    all_words = []
    word_id = 0
    for page_result in results:
        for word in page_result.get("words", []):
            all_words.append({
                "id": word_id,
                "content": word["content"],
                "points": _normalize_points(word["points"]),
                "direction": word.get("direction", "horizontal"),
            })
            word_id += 1

    full_text = " ".join(w["content"] for w in all_words)
    return {"text": full_text, "words": all_words, "word_count": len(all_words)}
