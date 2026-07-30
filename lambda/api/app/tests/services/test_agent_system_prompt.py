import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from decimal import Decimal

from services.agent_service import AgentService


class TestCreateSystemPrompt:
    def test_decimal_values_are_serializable(self):
        # DynamoDB から読んだ extracted_info は数値が Decimal になっている
        extracted_info = {"total": Decimal("1234.56"), "name": "sample"}
        prompt = AgentService._create_system_prompt(AgentService.__new__(AgentService), extracted_info)
        # json.dumps が例外を投げず、数値が float として埋め込まれる
        assert "1234.56" in prompt
        assert "sample" in prompt

    def test_nested_decimal_values(self):
        extracted_info = {"items": [{"price": Decimal("9.99")}]}
        prompt = AgentService._create_system_prompt(AgentService.__new__(AgentService), extracted_info)
        assert "9.99" in prompt
        # 埋め込まれた JSON 部分が再パース可能であること
        json_part = prompt.split("\n", 1)[1].strip()
        assert json.loads(json_part) == {"items": [{"price": 9.99}]}
