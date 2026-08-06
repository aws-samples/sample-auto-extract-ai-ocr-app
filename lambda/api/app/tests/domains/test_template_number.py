import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json

from domains.template import generate_json_template


class TestNumberFieldTemplate:
    def test_number_field_emits_numeric_hint_as_string(self):
        schema = {"fields": [{"name": "qty", "display_name": "数量", "type": "number"}]}
        template = generate_json_template(schema)
        # JSON string のまま（値を裸の数値にしない）で、数字のみのヒントが入る
        parsed = json.loads(template)
        assert isinstance(parsed["qty"], str)
        assert "数字のみ" in parsed["qty"]

    def test_string_field_unchanged(self):
        schema = {"fields": [{"name": "amount", "display_name": "金額", "type": "string"}]}
        parsed = json.loads(generate_json_template(schema))
        assert parsed["amount"] == "金額の値"

    def test_number_inside_list_of_map(self):
        schema = {"fields": [{
            "name": "items", "display_name": "明細", "type": "list",
            "items": {"type": "map", "fields": [{"name": "n", "display_name": "個数", "type": "number"}]},
        }]}
        parsed = json.loads(generate_json_template(schema))
        assert "数字のみ" in parsed["items"][0]["n"]

    def test_number_as_simple_list_item(self):
        schema = {"fields": [{
            "name": "codes", "display_name": "コード", "type": "list",
            "items": {"type": "number"},
        }]}
        parsed = json.loads(generate_json_template(schema))
        assert isinstance(parsed["codes"], list)
        assert "数字のみ" in parsed["codes"][0]
