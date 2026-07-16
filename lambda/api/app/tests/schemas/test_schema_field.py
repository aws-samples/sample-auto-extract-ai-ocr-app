import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from pydantic import ValidationError

from schemas.schema import SchemaField


class TestSchemaField:
    def test_valid_types_pass(self):
        for t in ("string", "number", "map", "list"):
            SchemaField(name="f", display_name="F", type=t)

    def test_nested_map_and_list(self):
        f = SchemaField(
            name="items", display_name="明細", type="list",
            items={"type": "map", "fields": [{"name": "d", "display_name": "品目", "type": "string"}]},
        )
        assert f.items.fields[0].name == "d"

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(name="f", display_name="F", type="bogus")

    def test_missing_name_rejected(self):
        with pytest.raises(ValidationError):
            SchemaField(display_name="F", type="string")

    def test_extra_key_rejected(self):
        # 定義外キー（typo・野良キー）は forbid で弾く
        with pytest.raises(ValidationError):
            SchemaField(name="f", display_name="F", type="string", feilds=[])
