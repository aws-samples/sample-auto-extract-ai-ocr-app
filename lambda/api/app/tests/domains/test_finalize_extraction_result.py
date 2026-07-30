import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from domains.extraction_engine import finalize_extraction_result


class TestFinalizeExtractionResult:
    def test_floats_are_returned_unchanged(self):
        # Decimal 変換は repository 層が担うため、ここでは float のまま返す
        extracted_info = {"total": 1234.56}
        result = finalize_extraction_result(extracted_info)
        assert result == {"extracted_info": {"total": 1234.56}}
        assert isinstance(result["extracted_info"]["total"], float)

    def test_mapping_included_when_present(self):
        result = finalize_extraction_result({"a": 1}, mapping={"a": [0]})
        assert result == {"extracted_info": {"a": 1}, "mapping": {"a": [0]}}

    def test_mapping_omitted_when_none(self):
        result = finalize_extraction_result({"a": 1})
        assert "mapping" not in result
