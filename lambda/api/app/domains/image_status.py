"""画像ステータス判定のドメインロジック"""


class ImageStatus:
    """画像の処理ステータス値。DynamoDB に生文字列で保存される。"""
    UPLOADING = "uploading"
    PENDING = "pending"
    CONVERTING = "converting"
    OCR = "ocr"
    EXTRACTING = "extracting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ALL = {UPLOADING, PENDING, CONVERTING, OCR, EXTRACTING, PROCESSING, COMPLETED, FAILED}


class AgentStatus:
    """エージェント検証のステータス値。"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ALL = {IDLE, PROCESSING, COMPLETED, FAILED, SKIPPED}


class PageProcessingMode:
    """複数ページ PDF の処理モード。"""
    COMBINED = "combined"
    INDIVIDUAL = "individual"
    ALL = {COMBINED, INDIVIDUAL}


# 処理中を表す status（OCR中・抽出中・汎用 processing）。親集約でまとめて扱う。
_PARENT_PROCESSING_STATUSES = {ImageStatus.OCR, ImageStatus.EXTRACTING, ImageStatus.PROCESSING}


def validate_image_status(status: str) -> None:
    """無効な画像ステータス値なら ValueError（write 前の検証用）。"""
    if status not in ImageStatus.ALL:
        raise ValueError(f"Invalid image status: {status!r}")


def validate_agent_status(status: str) -> None:
    """無効な agent_status 値なら ValueError。"""
    if status not in AgentStatus.ALL:
        raise ValueError(f"Invalid agent status: {status!r}")


def validate_page_processing_mode(mode: str) -> None:
    """無効な page_processing_mode 値なら ValueError。"""
    if mode not in PageProcessingMode.ALL:
        raise ValueError(f"Invalid page_processing_mode: {mode!r}")


def determine_parent_status(children: list[dict]) -> str:
    """子ページのステータスから親ドキュメントのステータスを判定する

    Args:
        children: 子ページのリスト

    Returns:
        親ドキュメントのステータス
    """
    if not children:
        return ImageStatus.CONVERTING

    statuses = [child.get("status") for child in children]

    if all(status == ImageStatus.COMPLETED for status in statuses):
        return ImageStatus.COMPLETED
    elif any(status == ImageStatus.FAILED for status in statuses):
        return ImageStatus.FAILED
    elif any(status in _PARENT_PROCESSING_STATUSES for status in statuses):
        return ImageStatus.PROCESSING
    else:
        return ImageStatus.CONVERTING


def determine_parent_agent_status(children: list[dict]) -> str:
    """子ページの agent_status から親ドキュメントの agent_status を判定する

    優先度: failed > processing > idle(未実行あり) > completed > skipped
    """
    if not children:
        return AgentStatus.IDLE

    statuses = [child.get("agent_status") or AgentStatus.IDLE for child in children]

    if any(s == AgentStatus.FAILED for s in statuses):
        return AgentStatus.FAILED
    if any(s == AgentStatus.PROCESSING for s in statuses):
        return AgentStatus.PROCESSING
    if any(s == AgentStatus.IDLE for s in statuses):
        return AgentStatus.PROCESSING
    if all(s == AgentStatus.COMPLETED for s in statuses):
        return AgentStatus.COMPLETED
    if all(s == AgentStatus.SKIPPED for s in statuses):
        return AgentStatus.SKIPPED
    return AgentStatus.COMPLETED
