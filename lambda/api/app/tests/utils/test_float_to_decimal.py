import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from decimal import Decimal

from utils.helpers import float_to_decimal


class TestFloatToDecimal:
    def test_float_becomes_decimal(self):
        assert float_to_decimal(1.5) == Decimal("1.5")
        assert isinstance(float_to_decimal(1.5), Decimal)

    def test_decimal_passthrough_is_idempotent(self):
        once = float_to_decimal(2.5)
        twice = float_to_decimal(once)
        assert once == twice == Decimal("2.5")
        assert isinstance(twice, Decimal)

    def test_int_passthrough(self):
        assert float_to_decimal(3) == 3
        assert isinstance(float_to_decimal(3), int)

    def test_bool_is_not_converted(self):
        assert float_to_decimal(True) is True
        assert float_to_decimal(False) is False

    def test_str_and_none_passthrough(self):
        assert float_to_decimal("abc") == "abc"
        assert float_to_decimal(None) is None

    def test_nested_dict_and_list(self):
        payload = {
            "amount": 100.5,
            "items": [{"price": 9.99}, {"price": 5}],
            "label": "invoice",
        }
        result = float_to_decimal(payload)
        assert result["amount"] == Decimal("100.5")
        assert result["items"][0]["price"] == Decimal("9.99")
        assert result["items"][1]["price"] == 5
        assert result["label"] == "invoice"

    def test_extracted_info_like_payload(self):
        payload = {"total": 1234.56, "customer": "sample"}
        result = float_to_decimal(payload)
        assert result["total"] == Decimal("1234.56")

    def test_suggestions_like_payload(self):
        suggestions = [
            {"original_value": 10.0, "suggested_value": 12.5, "field": "qty"},
        ]
        result = float_to_decimal(suggestions)
        assert result[0]["original_value"] == Decimal("10.0")
        assert result[0]["suggested_value"] == Decimal("12.5")
        assert result[0]["field"] == "qty"
