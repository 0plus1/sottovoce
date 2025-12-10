import importlib
import os
import unittest

from src import config


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        # Backup environment and clear relevant keys for each test.
        self._env_backup = os.environ.copy()
        baseline = {
            "RTSTT_MODEL": "base.en",
            "RTSTT_COMPUTE_TYPE": "default",
            "RTSTT_LANGUAGE": "",
            "RTSTT_USE_MICROPHONE": "true",
            "LLM_ENDPOINT": "http://localhost:1234/v1/chat/completions",
            "LLM_MODEL": "local-model",
            "LLM_TIMEOUT": "60",
            "SESSION_LOGS_DIR": "session_logs",
            "TTS_ENABLED": "",
            "TTS_VOICE_PATH": "",
            "TTS_USE_CUDA": "",
            "TTS_LENGTH_SCALE": "1.0",
            "TTS_NOISE_SCALE": "0.667",
            "TTS_NOISE_W_SCALE": "0.8",
            "TTS_VOLUME": "1.0",
        }
        for key, value in baseline.items():
            os.environ[key] = value

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
        self.assertFalse(settings.tts_enabled)
        self.assertEqual(settings.tts_voice_path, "")
        self.assertFalse(settings.tts_use_cuda)
        self.assertEqual(settings.tts_length_scale, 1.0)
        self.assertEqual(settings.tts_noise_scale, 0.667)
        self.assertEqual(settings.tts_noise_w_scale, 0.8)
        self.assertEqual(settings.tts_volume, 1.0)

    def test_env_overrides(self) -> None:
        os.environ["RTSTT_MODEL"] = "small"
        os.environ["RTSTT_COMPUTE_TYPE"] = "int8"
        os.environ["RTSTT_LANGUAGE"] = "en"
        os.environ["RTSTT_USE_MICROPHONE"] = "false"
        os.environ["LLM_ENDPOINT"] = "http://localhost:9000/v1/chat/completions"
        os.environ["LLM_MODEL"] = "lmstudio"
        os.environ["LLM_TIMEOUT"] = "15"
        os.environ["SESSION_LOGS_DIR"] = "logs"
        os.environ["TTS_ENABLED"] = "true"
        os.environ["TTS_VOICE_PATH"] = "/tmp/voice.onnx"
        os.environ["TTS_USE_CUDA"] = "true"
        os.environ["TTS_LENGTH_SCALE"] = "1.5"
        os.environ["TTS_NOISE_SCALE"] = "0.5"
        os.environ["TTS_NOISE_W_SCALE"] = "0.6"
        os.environ["TTS_VOLUME"] = "0.7"

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
        self.assertTrue(settings.tts_enabled)
        self.assertEqual(settings.tts_voice_path, "/tmp/voice.onnx")
        self.assertTrue(settings.tts_use_cuda)
        self.assertEqual(settings.tts_length_scale, 1.5)
        self.assertEqual(settings.tts_noise_scale, 0.5)
        self.assertEqual(settings.tts_noise_w_scale, 0.6)
        self.assertEqual(settings.tts_volume, 0.7)


if __name__ == "__main__":
    unittest.main()
