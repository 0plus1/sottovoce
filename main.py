
import numpy
import numpy.typing as numpy_types
from src.trascribe import transcribe # pyright: ignore[reportUnknownVariableType]
from src.recorder import record_audio_vad
from typing import Optional
from src.console import print_info, print_error, print_debug, print_success
from src.parameters import debug_mode

if __name__ == "__main__":
    print_info("Welcome to sottovoce!")
    print_info("Press Enter to start...\n")
    try:
        while True:
            print_info("[SYSTEM] Listening for speech...")
            audio_data: Optional[bytes] = record_audio_vad()
            if audio_data is None:
                print_error("[SYSTEM] No audio recorded. Please try again.")
                continue
            if (debug_mode):
                print_debug(f"[DEBUG] Returned from record_audio_vad, bytes recorded: {len(audio_data)}")
            audio_np: numpy_types.NDArray[numpy.float32] = numpy.frombuffer(audio_data, dtype=numpy.int16).astype(numpy.float32) / 32768.0
            if (debug_mode):
                print_debug(f"[DEBUG] audio_np shape: {audio_np.shape}, dtype: {audio_np.dtype}, size: {audio_np.size}")
            if audio_np.size == 0:
                print_error("[SYSTEM] No audio recorded. Please ensure your microphone is working.")
                continue
            print_info("[SYSTEM] Transcribing...")
            try:
                text: str = transcribe(audio_np)
                print_success(f"[SYSTEM] Transcribed: {text}")
            except Exception as e:
                print_error(f"[ERROR] Transcription failed: {e}")
            print_info("[SYSTEM] Listening cycle will restart.")
    except KeyboardInterrupt:
        print_info("\n[SYSTEM] Stopped by user. Goodbye!")
