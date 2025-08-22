import os
from dotenv import load_dotenv

load_dotenv() 

debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
whisper_fp16 = os.getenv("WHISPER_FP16", "false").lower() == "true"
whisper_model = os.getenv("WHISPER_MODEL", "base.en")
max_silence_ms = float(os.getenv("MAX_SILENCE_MS", "1000"))