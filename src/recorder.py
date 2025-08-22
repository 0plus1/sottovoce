import webrtcvad  # pyright: ignore[reportMissingTypeStubs]
import sounddevice  # pyright: ignore[reportMissingTypeStubs]
import time
from typing import Optional
from src.parameters import debug_mode, max_silence_ms
from src.console import print_debug, print_error
class VADState:
    def __init__(self):
        self.silence_count: int = 0
        self.started: bool = False
        self.should_stop: bool = False
        self.frames: list[bytes] = []

def record_audio_vad(
    aggressiveness: int = 2,
    sample_rate: int = 16000,
    frame_duration: int = 30,
    max_silence_ms: int = max_silence_ms
) -> Optional[bytes]:
    """
    Records audio from the microphone and stops when silence is detected using webrtcvad.
    Args:
        aggressiveness (int): VAD aggressiveness (0-3).
        sample_rate (int): Audio sample rate.
        frame_duration (int): Frame size in ms.
        max_silence_ms (int): How much silence (ms) before stopping.
    Returns:
        Optional[bytes]: Recorded audio as bytes, or None if no speech detected.
    """
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sample_rate * frame_duration / 1000)
    silence_threshold = max_silence_ms // frame_duration
    state = VADState()
    def callback(
        indata: memoryview,
        frame_count: int,
        time_: float,
        status: sounddevice.CallbackFlags
    ) -> None:
        frame_bytes: bytes = bytes(indata)
        is_speech: bool = vad.is_speech(frame_bytes, sample_rate)  # pyright: ignore
        if not state.started:
            if is_speech:
                state.started = True
                if debug_mode:
                    print_debug("[DEBUG] Started listening...")
            return
        state.frames.append(frame_bytes)
        if is_speech:
            state.silence_count = 0
        else:
            state.silence_count += 1
        if state.silence_count > silence_threshold:
            if debug_mode:
                print_debug("[SYSTEM] Stopped listening.")
                print_debug(f"[DEBUG] Silence threshold reached: {state.silence_count} > {silence_threshold}")
            state.should_stop = True

    try:
        with sounddevice.RawInputStream(
            samplerate=sample_rate,
            dtype="int16",
            channels=1,
            blocksize=frame_size,
            callback=callback
        ):
            while not state.should_stop:
                time.sleep(0.05)
    except Exception as e:
        print_error(f"[ERROR] Could not access audio device: {e}")
        return None
    if debug_mode:
        print_debug(f"[DEBUG] Number of frames recorded: {len(state.frames)}")
    if not state.frames:
        if debug_mode:
            print_debug("[SYSTEM] No speech detected. Please try again.")
        return None
    return b"".join(state.frames)

def list_devices() -> None:
    print("Available devices:")
    for idx, device in enumerate(sounddevice.query_devices()):
        print(f"{idx}: {device['name']} ({device['max_input_channels']} input channels)")