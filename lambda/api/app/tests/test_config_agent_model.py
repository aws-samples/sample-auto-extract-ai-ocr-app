"""エージェント検証用モデル設定のフォールバックのテスト。

Settings はクラス属性として import 時に評価されるため、環境変数を設定してから
モジュールを再読み込みして検証する。

想定している正しい挙動: AGENT_MODEL_ID / AGENT_MODEL_REGION は未設定または
空文字（CDK が未指定時に渡す値）のとき MODEL_ID / MODEL_REGION にフォールバックする。
"""
import importlib
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _reload_settings(env):
    with patch.dict(os.environ, env, clear=False):
        import config
        importlib.reload(config)
        return config.settings


def _restore_config():
    import config
    importlib.reload(config)


class TestAgentModelFallback:
    def teardown_method(self):
        _restore_config()

    def test_empty_env_falls_back_to_model_id(self):
        settings = _reload_settings({
            "MODEL_ID": "base-model",
            "MODEL_REGION": "us-west-2",
            "AGENT_MODEL_ID": "",
            "AGENT_MODEL_REGION": "",
        })
        assert settings.AGENT_MODEL_ID == "base-model"
        assert settings.AGENT_MODEL_REGION == "us-west-2"

    def test_explicit_agent_model_is_used(self):
        settings = _reload_settings({
            "MODEL_ID": "base-model",
            "MODEL_REGION": "us-west-2",
            "AGENT_MODEL_ID": "cheap-model",
            "AGENT_MODEL_REGION": "us-east-1",
        })
        assert settings.AGENT_MODEL_ID == "cheap-model"
        assert settings.AGENT_MODEL_REGION == "us-east-1"
