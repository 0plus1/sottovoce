import importlib
import os
import unittest

from src import config


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        # Backup environment and clear relevant keys for each test.
        self._env_backup = os.environ.copy()
        for key in (
            "RTSTT_MODEL",
            "RTSTT_COMPUTE_TYPE",
            "RTSTT_LANGUAGE",
            "RTSTT_USE_MICROPHONE",
            "LLM_ENDPOINT",
            "LLM_MODEL",
            "LLM_TIMEOUT",
            "SESSION_LOGS_DIR",
        ):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_defaults(self) -> None:
        importlib.reload(config)
        settings = config.get_settings()
        self.assertEqual(settings.rtstt_model, "base.en")
        self.assertEqual(settings.rtstt_compute_type, "default")
        self.assertEqual(settings.rtstt_language, "")
        self.assertTrue(settings.rtstt_use_microphone)
        self.assertEqual(settings.llm_endpoint, "http://localhost:1234/v1/chat/completions")
        self.assertEqual(settings.llm_model, "local-model")
        self.assertEqual(settings.llm_timeout, 60.0)
        self.assertEqual(settings.session_logs_dir, "session_logs")

    def test_env_overrides(self) -> None:
        os.environ["RTSTT_MODEL"] = "small"
        os.environ["RTSTT_COMPUTE_TYPE"] = "int8"
        os.environ["RTSTT_LANGUAGE"] = "en"
        os.environ["RTSTT_USE_MICROPHONE"] = "false"
        os.environ["LLM_ENDPOINT"] = "http://localhost:9000/v1/chat/completions"
        os.environ["LLM_MODEL"] = "lmstudio"
        os.environ["LLM_TIMEOUT"] = "15"
        os.environ["SESSION_LOGS_DIR"] = "logs"

        importlib.reload(config)
        settings = config.get_settings()
        self.assertEqual(settings.rtstt_model, "small")
        self.assertEqual(settings.rtstt_compute_type, "int8")
        self.assertEqual(settings.rtstt_language, "en")
        self.assertFalse(settings.rtstt_use_microphone)
        self.assertEqual(settings.llm_endpoint, "http://localhost:9000/v1/chat/completions")
        self.assertEqual(settings.llm_model, "lmstudio")
        self.assertEqual(settings.llm_timeout, 15.0)
        self.assertEqual(settings.session_logs_dir, "logs")


if __name__ == "__main__":
    unittest.main()
