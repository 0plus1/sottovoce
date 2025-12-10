from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy
import sounddevice as sd  # type: ignore[import-not-found]
from piper import PiperVoice, SynthesisConfig  # type: ignore[import-not-found]

from src.config import Settings


class TtsEngine:
    """Thin wrapper around Piper TTS."""

    def __init__(self, settings: Settings):
        self.enabled = settings.tts_enabled
        self.voice_path = settings.tts_voice_path
        self.use_cuda = settings.tts_use_cuda
        self.length_scale = settings.tts_length_scale
        self.noise_scale = settings.tts_noise_scale
        self.noise_w_scale = settings.tts_noise_w_scale
        self.volume = settings.tts_volume
        self._voice: Optional[PiperVoice] = None

    def _ensure_voice(self) -> None:
        if self._voice is None:
            if not self.voice_path:
                raise RuntimeError("TTS voice path is not set. Set TTS_VOICE_PATH to a Piper voice .onnx.")
            self._voice = PiperVoice.load(self.voice_path, use_cuda=self.use_cuda)

    def _synth_config(self):
        assert self._voice is not None
        return SynthesisConfig(
            volume=self.volume,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
        )

    def synthesize(self, text: str, output_path: Path) -> Path:
        if not self.enabled:
            raise RuntimeError("TTS is disabled")
        self._ensure_voice()
        assert self._voice is not None
        syn_config = self._synth_config()

        # Play back to the user (no disk write)
        chunks = list(self._voice.synthesize(text, syn_config=syn_config))
        if chunks:
            sample_rate = chunks[0].sample_rate
            channels = chunks[0].sample_channels
            audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in chunks)
            audio = numpy.frombuffer(audio_bytes, dtype=numpy.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels)
            sd.play(audio, sample_rate)
            sd.wait()

        return output_path
