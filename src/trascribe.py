import numpy
import numpy.typing as npt
import whisper  # pyright: ignore[reportMissingTypeStubs]
from typing import Dict, Any
from src.parameters import whisper_fp16

def transcribe(audio_np: npt.NDArray[numpy.float32]) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.
    """
    stt: whisper.Whisper = whisper.load_model("base.en")
    result: Dict[str, Any] = stt.transcribe(audio_np, fp16=whisper_fp16) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    text: str = result["text"].strip()
    return text