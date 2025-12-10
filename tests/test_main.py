import unittest
from typing import Optional

from src.config import Settings
import main


class DummyRecorder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def text(self) -> str:
        return "dummy"

    def __enter__(self) -> "DummyRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class MainTests(unittest.TestCase):
    def test_build_recorder_passes_settings(self) -> None:
        settings = Settings(
            model="tiny",
            compute_type="int8",
            language="en",
            use_microphone=False,
            log_level=10,
            no_log_file=True,
        )

        original_cls = main.AudioToTextRecorder
        try:
            main.AudioToTextRecorder = DummyRecorder  # type: ignore[assignment]
            recorder = main.build_recorder(settings)
        finally:
            main.AudioToTextRecorder = original_cls

        self.assertIsInstance(recorder, DummyRecorder)
        self.assertEqual(recorder.kwargs["model"], "tiny")
        self.assertEqual(recorder.kwargs["compute_type"], "int8")
        self.assertEqual(recorder.kwargs["language"], "en")
        self.assertFalse(recorder.kwargs["use_microphone"])
        self.assertEqual(recorder.kwargs["level"], 10)


if __name__ == "__main__":
    unittest.main()
