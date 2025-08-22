import pytest
from src.recorder import record_audio_vad, list_devices, VADState

# Test VADState class

def test_vadstate_initialization():
    state = VADState()
    assert state.silence_count == 0
    assert state.started is False
    assert state.should_stop is False
    assert isinstance(state.frames, list)
    assert state.frames == []

# Test list_devices (should not raise)
def test_list_devices_runs():
    try:
        list_devices()
    except Exception as e:
        pytest.fail(f"list_devices raised an exception: {e}")

# Test record_audio_vad (mocked)
def fake_raw_input_stream(*args, **kwargs):
    class DummyStream:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    return DummyStream()

def test_record_audio_vad_no_audio(monkeypatch):
    # Patch sounddevice.RawInputStream to simulate no audio
    import src.recorder as recorder
    monkeypatch.setattr(recorder.sounddevice, "RawInputStream", fake_raw_input_stream)
    # Patch VADState to simulate no frames
    class DummyVADState:
        def __init__(self):
            self.silence_count = 0
            self.started = True
            self.should_stop = True
            self.frames = []
    monkeypatch.setattr(recorder, "VADState", DummyVADState)
    result = record_audio_vad()
    assert result is None

def test_record_audio_vad_with_audio(monkeypatch):
    # Patch sounddevice.RawInputStream to simulate audio
    import src.recorder as recorder
    monkeypatch.setattr(recorder.sounddevice, "RawInputStream", fake_raw_input_stream)
    # Patch VADState to simulate frames
    class DummyVADState:
        def __init__(self):
            self.silence_count = 0
            self.started = True
            self.should_stop = True
            self.frames = [b'1234', b'5678']
    monkeypatch.setattr(recorder, "VADState", DummyVADState)
    result = record_audio_vad()
    assert isinstance(result, bytes)
    assert result == b'12345678'
