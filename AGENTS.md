# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a Python-based Local AI Home Assistant that combines voice input (Whisper STT + Silero VAD), camera vision (OpenCV), LLM intelligence (OpenAI GPT-4o), and TTS output (ElevenLabs). The project is currently unfinished — there is no unified `main.py` entry point. See `README.md` and `docs/AI_Custom_Voice_Personality_Guide.md` for full context.

### Environment setup

- **Python venv** lives at `/workspace/venv`. Activate with `source /workspace/venv/bin/activate`.
- System packages `python3.12-venv`, `python3-dev`, `portaudio19-dev`, and `ffmpeg` are required (pre-installed in snapshot).
- All Python deps (required + optional from `requirements.txt`) are installed in the venv, including `openai`, `elevenlabs`, `opencv-python`, `mediapipe`, and `pydub`.

### Running source modules

- The only runnable entry point is `src/audio/transcribe_audio.py` (`python src/audio/transcribe_audio.py` from the repo root), which runs a record-and-transcribe loop. It requires a microphone, so it will not work in a headless cloud environment.
- To import `src/audio/transcribe_audio.py`, you must add `src/audio` to `sys.path` because it imports `voice_recorder` as a bare module (no `__init__.py` or package structure).
- `src/llm/openai.py` has a circular import issue: the file is named `openai.py` which shadows the `openai` package. Import it via `importlib.util.spec_from_file_location` or rename the file if editing is allowed.

### Gotchas

- **No hardware in cloud**: Microphone (`sounddevice`), camera (`cv2.VideoCapture`), and speaker playback will fail in a headless environment. Test with synthetic numpy arrays and `frame_to_base64` instead.
- **No tests or linter**: The project has no test suite, no linting config (pylint/flake8/ruff), and no `pyproject.toml`. Linting and testing are limited to import verification and manual functional checks.
- **API keys**: `OPENAI_API_KEY` and `ELEVENLABS_API_KEY` are needed for LLM and TTS modules respectively. Without them, those modules will error at runtime (not at import time for OpenAI, but at call time).
- **Whisper model download**: The Whisper `base` model (~139 MB) is downloaded automatically on first load. It's cached in `~/.cache/whisper/`.
