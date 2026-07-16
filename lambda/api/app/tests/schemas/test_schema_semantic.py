import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from pydantic import ValidationError

from schemas.schema import SchemaField, SchemaSaveRequest


def _save_request(fields):
    return SchemaSaveRequest(
        name="app1", display_name="App 1", fields=fields,
        input_methods={"file_upload": True},
    )


def _str_field(name="f", display_name="F"):
    return {"name": name, "display_name": display_name, "type": "string"}


class TestRule1NonEmptySchema:
    def test_empty_fields_rejected(self):
        with pytest.raises(ValidationError):
            _save_request([])

    def test_one_field_ok(self):
        _save_request([_str_field()])


class TestRule2FieldName:
    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="", display_name="F", type="string")

    @pytest.mark.parametrize("bad", ["a b", "日本語", "field@1", "a-b", "a.b"])
    def test_invalid_pattern_rejected(self, bad):
        with pytest.raises(ValidationError):
            SchemaField(name=bad, display_name="F", type="string")

    @pytest.mark.parametrize("ok", ["a", "field_1", "ABC_123", "_x"])
    def test_valid_pattern_ok(self, ok):
        SchemaField(name=ok, display_name="F", type="string")

    def test_nested_map_child_name_validated(self):
        with pytest.raises(ValidationError):
            SchemaField(name="m", display_name="M", type="map",
                        fields=[{"name": "bad name", "display_name": "C", "type": "string"}])


class TestRule3SiblingUnique:
    def test_top_level_duplicate_rejected(self):
        with pytest.raises(ValidationError):
            _save_request([_str_field(name="a"), _str_field(name="a")])

    def test_same_name_different_parent_ok(self):
        # 親が違えば同名は許可（グローバル一意ではない）
        _save_request([
            {"name": "m1", "display_name": "M1", "type": "map",
             "fields": [_str_field(name="x")]},
            {"name": "m2", "display_name": "M2", "type": "map",
             "fields": [_str_field(name="x")]},
        ])

    def test_map_child_duplicate_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="m", display_name="M", type="map",
                        fields=[_str_field(name="a"), _str_field(name="a")])


class TestRule4Structure:
    def test_map_without_fields_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="m", display_name="M", type="map")

    def test_map_with_empty_fields_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="m", display_name="M", type="map", fields=[])

    def test_map_with_items_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="m", display_name="M", type="map",
                        fields=[_str_field()], items={"type": "string"})

    def test_list_without_items_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="l", display_name="L", type="list")

    def test_list_with_fields_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="l", display_name="L", type="list",
                        items={"type": "string"}, fields=[_str_field()])

    def test_string_with_fields_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="s", display_name="S", type="string", fields=[_str_field()])

    def test_number_with_items_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="n", display_name="N", type="number", items={"type": "string"})

    def test_list_item_type_list_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="l", display_name="L", type="list", items={"type": "list"})

    def test_list_item_map_without_fields_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="l", display_name="L", type="list", items={"type": "map"})

    def test_valid_list_of_map_ok(self):
        # 明細行の典型パターン（list -> map -> string）は通る
        f = SchemaField(name="items", display_name="明細", type="list",
                        items={"type": "map",
                               "fields": [_str_field(name="d", display_name="品目")]})
        assert f.items.fields[0].name == "d"


class TestRule5DisplayName:
    def test_empty_display_name_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="f", display_name="", type="string")

    def test_whitespace_display_name_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="f", display_name="   ", type="string")
