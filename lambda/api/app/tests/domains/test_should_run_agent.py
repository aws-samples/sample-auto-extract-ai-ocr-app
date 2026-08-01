import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from domains.schema_fields import should_run_agent


class TestShouldRunAgent:
    def test_auto_run_requires_enabled_and_auto_run(self):
        assert should_run_agent({"agent_enabled": True, "agent_auto_run": True}) is True
        assert should_run_agent({"agent_enabled": True, "agent_auto_run": False}) is False
        assert should_run_agent({"agent_enabled": False, "agent_auto_run": True}) is False

    def test_manual_requires_only_enabled(self):
        assert should_run_agent({"agent_enabled": True, "agent_auto_run": False}, manual=True) is True
        assert should_run_agent({"agent_enabled": False, "agent_auto_run": True}, manual=True) is False

    def test_none_schema_is_false(self):
        assert should_run_agent(None) is False
        assert should_run_agent(None, manual=True) is False

    def test_missing_keys_default_false(self):
        assert should_run_agent({}) is False
        assert should_run_agent({}, manual=True) is False
