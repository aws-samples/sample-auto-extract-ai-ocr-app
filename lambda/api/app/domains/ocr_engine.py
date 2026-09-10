"""OCR エンジン — 純粋な OCR レスポンス整形ロジック

SageMaker / S3 への直接アクセスは行わない。
外部サービス呼び出しは ocr_service.py が担当する。
"""
import logging
import math
from typing import Any

from exceptions import ResponseParseError
from utils.bedrock import extract_json_from_response

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


def _box_to_points(box: Any, image_width: int, image_height: int) -> list | None:
    """LLM が出力した外接矩形 [x_min, y_min, x_max, y_max] を四隅の points に変換する。

    座標は画像範囲 (0〜幅, 0〜高さ) にクランプし int に丸める。
    有限数 4 要素でない場合や、クランプ後に矩形が成立しない（反転・面積ゼロ）場合は
    None を返す。
    """
    if not isinstance(box, list) or len(box) != 4:
        return None

    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None

    x_min = int(round(min(max(values[0], 0), image_width)))
    y_min = int(round(min(max(values[1], 0), image_height)))
    x_max = int(round(min(max(values[2], 0), image_width)))
    y_max = int(round(min(max(values[3], 0), image_height)))

    if x_min >= x_max or y_min >= y_max:
        return None

    # 左上から時計回りの四隅（既存 OCR エンジンの points 形式に合わせる）
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def parse_llm_ocr_response(response_text: str, image_width: int, image_height: int) -> dict:
    """LLM OCR のレスポンステキストを既存 OCR 結果形式に変換する（純粋関数）

    期待する応答: {"words": [{"text": str, "box": [x_min, y_min, x_max, y_max]}, ...]}
    box が不正な word は points なしで含める（OcrWord.points は Optional）。

    Raises:
        ResponseParseError: 応答から words リストの JSON を読み取れなかった場合
    """
    parsed = extract_json_from_response(response_text)
    raw_words = parsed.get("words") if isinstance(parsed, dict) else None
    if not isinstance(raw_words, list):
        raise ResponseParseError("LLM OCR 応答から words リストを読み取れませんでした")

    words = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            continue
        content = str(raw_word.get("text", "")).strip()
        if not content:
            continue

        word = {"id": len(words), "content": content}
        points = _box_to_points(raw_word.get("box"), image_width, image_height)
        if points:
            word["points"] = points
        words.append(word)

    full_text = " ".join(word["content"] for word in words)
    return {"text": full_text, "words": words, "word_count": len(words)}


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
