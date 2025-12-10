# SOTTOVOCE
Private, local LLM based emotional support.

[![Coverage Status](https://coveralls.io/repos/github/0plus1/sottovoce/badge.svg?branch=main)](https://coveralls.io/github/0plus1/sottovoce?branch=main)

> **Disclaimer:**  
> SOTTOVOCE is not a certified therapist or a substitute for professional mental health care. If you are experiencing a crisis or need immediate help, please reach out to a qualified professional or contact a local helpline.

For a list of suicide prevention hotlines and mental health resources worldwide, visit:  
- [Befrienders Worldwide](https://www.befrienders.org/)
- [International Association for Suicide Prevention (IASP) - Crisis Centres](https://www.iasp.info/crisis-centres-helplines/)
- [Suicide.org - International Suicide Hotlines](https://www.suicide.org/international-suicide-hotlines.html)

## Dependencies
This project uses:
* [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) for the main voice loop
* [Piper](https://github.com/OHF-Voice/piper1-gpl/tree/main) for speech synthesis

## Install

### System wide requirements
Install [ffmpeg](https://www.ffmpeg.org/)
Install [uv](https://github.com/astral-sh/uv)

### Repo setup
Install dependencies: `uv sync`

Copy `.env-default` to `.env`

Download a voice and make sure it's in the `voices` folder.
The repo is pre-configured to work out of the box with this command:
```
uv run -m piper.download_voices en_US-lessac-medium --data-dir ./voices
```

Create an optional `PROMPT.md` file, containing the system prompt.

## Run

```sh
uv run ./main.py
```
Loop: listen → transcribe → send to local LLM → log turn → repeat. Session logs live in `session_logs/`.
Press Ctrl+C to exit.

## Test
```
coverage run -m pytest -p no:warnings
# python -m dotenv run -- coveralls
```