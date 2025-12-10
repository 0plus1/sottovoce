import unittest
from unittest.mock import patch, MagicMock

from src.config import Settings
import main


class DummyRecorder:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def text(self) -> str:
        return "dummy"

    def __enter__(self) -> "DummyRecorder":
        return self

    def __exit__(self) -> None:
        return None


class MainTests(unittest.TestCase):
    def test_build_recorder_passes_settings(self) -> None:
        settings = Settings(
            rtstt_model="tiny",
            rtstt_compute_type="int8",
            rtstt_language="en",
            rtstt_use_microphone=False,
            llm_endpoint="http://localhost:1234/v1/chat/completions",
            llm_model="test-model",
            llm_timeout=5.0,
            session_logs_dir="session_logs",
            tts_enabled=True,
            tts_voice_path="/tmp/voice.onnx",
            tts_use_cuda=False,
            tts_length_scale=1.0,
            tts_noise_scale=0.667,
            tts_noise_w_scale=0.8,
            tts_volume=1.0,
        )

        original_cls = main.AudioToTextRecorder
        try:
            main.AudioToTextRecorder = DummyRecorder  # type: ignore[assignment]
            recorder = main.build_recorder(settings)
        finally:
            main.AudioToTextRecorder = original_cls

        self.assertIsInstance(recorder, DummyRecorder)
        self.assertEqual(recorder.kwargs["model"], "tiny")  # type: ignore[attr-defined]
        self.assertEqual(recorder.kwargs["compute_type"], "int8")  # type: ignore[attr-defined]
        self.assertEqual(recorder.kwargs["language"], "en")  # type: ignore[attr-defined]
        self.assertFalse(recorder.kwargs["use_microphone"])  # type: ignore[attr-defined]

    def test_build_llm_client(self) -> None:
        settings = Settings(
            rtstt_model="tiny",
            rtstt_compute_type="int8",
            rtstt_language="en",
            rtstt_use_microphone=False,
            llm_endpoint="http://localhost:9999/v1/chat/completions",
            llm_model="local-model",
            llm_timeout=7.5,
            session_logs_dir="session_logs",
            tts_enabled=False,
            tts_voice_path="",
            tts_use_cuda=False,
            tts_length_scale=1.0,
            tts_noise_scale=0.667,
            tts_noise_w_scale=0.8,
            tts_volume=1.0,
        )
        client = main.build_llm_client(settings)
        self.assertEqual(client.config.endpoint, "http://localhost:9999/v1/chat/completions")
        self.assertEqual(client.config.model, "local-model")
        self.assertEqual(client.config.timeout, 7.5)

    def test_llm_client_system_prompt_injection(self) -> None:
        settings = Settings(
            rtstt_model="tiny",
            rtstt_compute_type="int8",
            rtstt_language="en",
            rtstt_use_microphone=False,
            llm_endpoint="http://localhost:9999/v1/chat/completions",
            llm_model="local-model",
            llm_timeout=7.5,
            session_logs_dir="session_logs",
            tts_enabled=False,
            tts_voice_path="",
            tts_use_cuda=False,
            tts_length_scale=1.0,
            tts_noise_scale=0.667,
            tts_noise_w_scale=0.8,
            tts_volume=1.0,
        )
        client = main.build_llm_client(settings)
        client.system_prompt = "system here"

        fake_response = MagicMock()
        fake_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        fake_response.raise_for_status.return_value = None

        with patch("src.llm_client.requests.post", return_value=fake_response) as mock_post:
            result = client.complete("user msg")

        self.assertEqual(result, "ok")
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "system here")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "user msg")


if __name__ == "__main__":
    unittest.main()
