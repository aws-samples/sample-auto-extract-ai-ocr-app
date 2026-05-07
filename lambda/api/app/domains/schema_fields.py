"""スキーマフィールドのドメインロジック"""


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
