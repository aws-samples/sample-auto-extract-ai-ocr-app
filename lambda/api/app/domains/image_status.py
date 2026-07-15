"""画像ステータス判定のドメインロジック"""

# OCR/抽出フェーズを区別する内部 status。フロントには公開しない。
_INTERNAL_PHASES = {"ocr", "extracting"}
# 親ドキュメントを processing とみなす子ステータス（内部フェーズ + processing）。
_PARENT_PROCESSING_STATUSES = _INTERNAL_PHASES | {"processing"}


def to_api_status(status: str | None) -> str | None:
    """内部 status を API レスポンス用の status に畳む

    内部では OCR/抽出フェーズを ocr/extracting で区別するが、フロントは
    processing/completed/failed で動くため、境界でここを通す。
    """
    if status in _INTERNAL_PHASES:
        return "processing"
    return status


def determine_parent_status(children: list[dict]) -> str:
    """子ページのステータスから親ドキュメントのステータスを判定する

    Args:
        children: 子ページのリスト

    Returns:
        親ドキュメントのステータス
    """
    if not children:
        return "converting"

    statuses = [child.get("status") for child in children]

    if all(status == "completed" for status in statuses):
        return "completed"
    elif any(status == "failed" for status in statuses):
        return "failed"
    elif any(status in _PARENT_PROCESSING_STATUSES for status in statuses):
        return "processing"
    else:
        return "converting"


def determine_parent_agent_status(children: list[dict]) -> str:
    """子ページの agent_status から親ドキュメントの agent_status を判定する

    優先度: failed > processing > idle(未実行あり) > completed > skipped
    """
    if not children:
        return "idle"

    statuses = [child.get("agent_status") or "idle" for child in children]

    if any(s == "failed" for s in statuses):
        return "failed"
    if any(s == "processing" for s in statuses):
        return "processing"
    if any(s == "idle" for s in statuses):
        return "processing"
    if all(s == "completed" for s in statuses):
        return "completed"
    if all(s == "skipped" for s in statuses):
        return "skipped"
    return "completed"
