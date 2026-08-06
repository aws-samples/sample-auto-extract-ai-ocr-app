"""スキーマ定義から抽出対象のフィールド名を平坦化する処理のテスト。

ここで作られる名前は LLM に渡すテンプレートと OCR 語との対応表のキーになるため、
階層の表し方が変わると抽出結果の突き合わせが壊れる。

想定している正しい挙動:
- 入れ子（map）はドット記法で親名を前置して再帰的に展開する。
- 表（list of map）は行数が不定なので `親.項目` の形で列名までを出す。
- 値の並び（list of scalar）は個々の要素に名前が無いので親名だけを出す。
- 親自身の名前も必ず含める（親だけを指す指示も成立させるため）。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from domains.schema_fields import extract_field_names


class TestExtractFieldNames:
    def test_flat_fields(self):
        fields = [
            {"name": "issuer", "type": "string"},
            {"name": "total", "type": "number"},
        ]
        assert extract_field_names(fields) == ["issuer", "total"]

    def test_map_is_expanded_with_dot_notation(self):
        fields = [{
            "name": "sender",
            "type": "map",
            "fields": [
                {"name": "company", "type": "string"},
                {"name": "tel", "type": "string"},
            ],
        }]
        assert extract_field_names(fields) == ["sender", "sender.company", "sender.tel"]

    def test_nested_map_is_expanded_recursively(self):
        fields = [{
            "name": "sender",
            "type": "map",
            "fields": [{
                "name": "address",
                "type": "map",
                "fields": [{"name": "city", "type": "string"}],
            }],
        }]
        assert extract_field_names(fields) == [
            "sender", "sender.address", "sender.address.city",
        ]

    def test_list_of_map_yields_column_names(self):
        fields = [{
            "name": "items",
            "type": "list",
            "items": {
                "type": "map",
                "fields": [
                    {"name": "label", "type": "string"},
                    {"name": "amount", "type": "number"},
                ],
            },
        }]
        assert extract_field_names(fields) == ["items", "items.label", "items.amount"]

    def test_list_of_scalar_yields_only_parent_name(self):
        fields = [{
            "name": "notes",
            "type": "list",
            "items": {"type": "string"},
        }]
        assert extract_field_names(fields) == ["notes"]

    def test_empty_fields(self):
        assert extract_field_names([]) == []
