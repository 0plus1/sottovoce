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
            "RTSTT_LOG_LEVEL",
        ):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_defaults(self) -> None:
        importlib.reload(config)
        settings = config.get_settings()
        self.assertEqual(settings.model, "base.en")
        self.assertEqual(settings.compute_type, "default")
        self.assertEqual(settings.language, "")
        self.assertTrue(settings.use_microphone)
        # Default log level is WARNING
        self.assertEqual(settings.log_level, 30)

    def test_env_overrides(self) -> None:
        os.environ["RTSTT_MODEL"] = "small"
        os.environ["RTSTT_COMPUTE_TYPE"] = "int8"
        os.environ["RTSTT_LANGUAGE"] = "en"
        os.environ["RTSTT_USE_MICROPHONE"] = "false"
        os.environ["RTSTT_LOG_LEVEL"] = "INFO"

        importlib.reload(config)
        settings = config.get_settings()
        self.assertEqual(settings.model, "small")
        self.assertEqual(settings.compute_type, "int8")
        self.assertEqual(settings.language, "en")
        self.assertFalse(settings.use_microphone)
        self.assertEqual(settings.log_level, 20)
        self.assertFalse(settings.no_log_file)


if __name__ == "__main__":
    unittest.main()
