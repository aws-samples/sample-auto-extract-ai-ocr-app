"""
テンプレート生成の共通ユーティリティ
OCRとExtractionで共通して使用されるテンプレート生成機能を提供
"""

import json
import logging

logger = logging.getLogger(__name__)


def generate_unified_template(schema) -> str:
    """
    抽出データとindicesを含む統合されたJSONテンプレートを生成

    Args:
        schema: アプリケーションスキーマ（辞書またはリスト形式）

    Returns:
        str: 統合されたJSONテンプレート文字列
    """
    # 既存の関数を使用してテンプレートを生成
    json_template = generate_json_template(schema)
    indices_template = generate_indices_template(schema)

    # 統合されたテンプレートを作成
    unified_template = f"""{{
  "extracted_data": {json_template},
  "indices": {indices_template}
}}"""

    return unified_template


def generate_json_template(schema) -> str:
    """
    スキーマからJSONテンプレートを生成

    Args:
        schema: アプリケーションスキーマ（辞書またはリスト形式）

    Returns:
        str: JSONテンプレート文字列
    """
    def generate_field_template(fields, indent=2):
        items = []
        indent_str = " " * indent

        for field in fields:
            # fieldが辞書でない場合はスキップ
            if not isinstance(field, dict):
                logger.warning(f"フィールドが辞書形式ではありません: {type(field)} - {field}")
                continue

            if field.get("type") == "string":
                items.append(
                    f'{indent_str}"{field["name"]}": "{field["display_name"]}の値"')
            elif field.get("type") == "map" and "fields" in field:
                nested_template = generate_field_template(
                    field["fields"], indent + 2)
                items.append(
                    f'{indent_str}"{field["name"]}": {{\n{nested_template}\n{indent_str}}}')
            elif field.get("type") == "list" and "items" in field:
                if field["items"].get("type") == "map":
                    nested_template = generate_field_template(
                        field["items"]["fields"], indent + 4)
                    items.append(
                        f'{indent_str}"{field["name"]}": [\n{indent_str}  {{\n{nested_template}\n{indent_str}  }}\n{indent_str}]')
                else:
                    items.append(
                        f'{indent_str}"{field["name"]}": ["{field["display_name"]}の値"]')
            else:
                items.append(
                    f'{indent_str}"{field["name"]}": "{field["display_name"]}の値"')

        return ",\n".join(items)

    # schemaの型をチェック
    if isinstance(schema, dict):
        # 辞書形式の場合
        fields = schema.get("fields", [])
    elif isinstance(schema, list):
        # リスト形式の場合（fieldsが直接渡された場合）
        fields = schema
    else:
        logger.error(f"スキーマが予期しない形式です: {type(schema)} - {schema}")
        fields = []

    return "{\n" + generate_field_template(fields) + "\n}"


def generate_indices_template(schema) -> str:
    """
    スキーマからindicesテンプレートを生成（マッピング用）

    Args:
        schema: アプリケーションスキーマ（辞書またはリスト形式）

    Returns:
        str: indicesテンプレート文字列
    """
    def generate_indices_fields(fields, indent=4):
        items = []
        indent_str = " " * indent

        for field in fields:
            # fieldが辞書でない場合はスキップ
            if not isinstance(field, dict):
                logger.warning(f"フィールドが辞書形式ではありません: {type(field)} - {field}")
                continue

            if field.get("type") == "string":
                items.append(f'{indent_str}"{field["name"]}": [対応する単語のID]')
            elif field.get("type") == "map" and "fields" in field:
                nested_indices = generate_indices_fields(
                    field["fields"], indent + 2)
                items.append(
                    f'{indent_str}"{field["name"]}": {{\n{nested_indices}\n{indent_str}}}')
            elif field.get("type") == "list" and "items" in field:
                if field["items"].get("type") == "map":
                    nested_indices = []
                    for item_field in field["items"]["fields"]:
                        if isinstance(item_field, dict):
                            nested_indices.append(
                                f'{indent_str}      "{item_field["name"]}": [対応する単語のID]')
                    nested_indices_str = ",\n".join(nested_indices)
                    items.append(
                        f'{indent_str}"{field["name"]}": [\n{indent_str}  {{\n{nested_indices_str}\n{indent_str}  }}\n{indent_str}]')
                else:
                    items.append(
                        f'{indent_str}"{field["name"]}": [[対応する単語のID]]')
            else:
                items.append(f'{indent_str}"{field["name"]}": [対応する単語のID]')

        return ",\n".join(items)

    # schemaの型をチェック
    if isinstance(schema, dict):
        # 辞書形式の場合
        fields = schema.get("fields", [])
    elif isinstance(schema, list):
        # リスト形式の場合（fieldsが直接渡された場合）
        fields = schema
    else:
        logger.error(f"スキーマが予期しない形式です: {type(schema)} - {schema}")
        fields = []

    return '"indices": {\n' + generate_indices_fields(fields) + '\n}'
