# SOTTOVOCE
Private, local LLM based conversational partner, with long term memory.
[![Coverage Status](https://coveralls.io/repos/github/0plus1/sottovoce/badge.svg?branch=main)](https://coveralls.io/github/0plus1/sottovoce?branch=main)

> **Disclaimer:**  
> SOTTOVOCE is not a certified therapist or a substitute for professional mental health care.
If you are experiencing a crisis or need immediate help, please reach out to a qualified professional or contact a local helpline.

For a list of suicide prevention hotlines and mental health resources worldwide, visit:  
- [Befrienders Worldwide](https://www.befrienders.org/)
- [International Association for Suicide Prevention (IASP) - Crisis Centres](https://www.iasp.info/crisis-centres-helplines/)
- [Suicide.org - International Suicide Hotlines](https://www.suicide.org/international-suicide-hotlines.html)

## Dependencies
This project uses:
* [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) for the conversation loop
* [Piper](https://github.com/OHF-Voice/piper1-gpl/tree/main) for speech synthesis
* [LangChain](https://python.langchain.com/) for recent-context memory persisted locally

## Install

### System wide requirements
* [ffmpeg](https://www.ffmpeg.org/)
* [uv](https://github.com/astral-sh/uv)

### Repo setup
Install dependencies: `uv sync`

Copy `.env-default` to `.env`

Download a voice and make sure it's in the `voices` folder.
The repo is pre-configured to work out of the box with this command:
```
uv run -m piper.download_voices en_US-lessac-medium --data-dir ./voices
```
You can use any voice you like, simply change the environment variable `TTS_VOICE_PATH`

Create an optional `PROMPT.md` file, containing the system prompt.

## Run

```sh
uv run ./main.py
```

## How it works (high level)
- RealtimeSTT handles microphone VAD + transcription.
- LLM replies come from your local LM Studio-compatible endpoint, guided by `PROMPT.md` (system prompt).
- LangChain keeps recent context: a small rolling window of messages is persisted in a local SQLite file (`memory/memory.db`) and injected into each LLM call; token usage is monitored to respect a context window.
- If context window limit is approaching the conversation is summarised to allow near unlimited exchanges.
- Responses are logged to `session_logs/` and optionally spoken via Piper TTS if enabled.

## Test
```
uv run coverage run -m pytest -p no:warnings
# uv run dotenv run -- coveralls
```
