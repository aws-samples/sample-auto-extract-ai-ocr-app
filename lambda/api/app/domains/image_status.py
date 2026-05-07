"""画像ステータス判定のドメインロジック"""


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
    elif any(status == "processing" for status in statuses):
        return "processing"
    else:
        return "converting"
