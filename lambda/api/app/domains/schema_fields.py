"""スキーマフィールドのドメインロジック"""


def should_run_agent(schema: dict | None, manual: bool = False) -> bool:
    """このユースケースで Agent 検証を実行すべきか判定する。

    手動実行は `agent_enabled` のみ要求する。自動実行（抽出後の連動）は
    `agent_enabled` かつ `agent_auto_run` の両方が必要。

    Args:
        schema: app スキーマ（None 可）
        manual: 手動実行なら True

    Returns:
        検証を実行すべきなら True
    """
    enabled = bool(schema and schema.get("agent_enabled", False))
    if manual:
        return enabled
    return enabled and bool(schema and schema.get("agent_auto_run", False))


def extract_field_names(fields: list[dict], prefix: str = "") -> list[str]:
    """フィールド定義から階層構造を考慮したフィールド名リストを生成する。

    Args:
        fields: フィールド定義のリスト
        prefix: 親フィールドのプレフィックス

    Returns:
        フラット化されたフィールド名のリスト
    """
    names = []
    for field in fields:
        full_name = f"{prefix}{field['name']}" if prefix else field["name"]
        names.append(full_name)

        if field.get("type") == "map" and "fields" in field:
            names.extend(extract_field_names(field["fields"], f"{full_name}."))

        if field.get("type") == "list" and "items" in field:
            items = field["items"]
            if items.get("type") == "map" and "fields" in items:
                for item_field in items["fields"]:
                    names.append(f"{full_name}.{item_field['name']}")

    return names
