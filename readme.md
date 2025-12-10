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

## Install

1) Install `ffmpeg`
2) Install `uv`
3) Install deps: `uv sync`
4) Copy `.env-default` to `.env`

## Run

```sh
uv run ./main.py
```
Loop: listen → transcribe → repeat. Press Ctrl+C to exit.

## Test
```
coverage run -m pytest
```