THIS PROJECT IS CURRENTLY UNFINISHED!

The idea of this project is to allow the user to set up a Local AI Home Assistant.
Users can choose third-party services for LLM and TTS features, including OpenAI and
ElevenLabs.

## API key configuration

This project now supports a local configuration file and browser-based settings page.

- Runtime config file: `config/settings.json` (created automatically from `config/settings.example.json`)
- Settings web UI: `src/config/settings_web.py`

### Run the settings website

1. Activate the virtual environment:
   - `source /workspace/venv/bin/activate`
2. Start the local settings server:
   - `python src/config/settings_web.py --host 127.0.0.1 --port 8765`
3. Open your browser to:
   - `http://127.0.0.1:8765`
4. Enter and save:
   - OpenAI API key
   - ElevenLabs API key

Saved keys are read by:

- `src/llm/openai.py`
- `src/tts/elevenlabs.py`

If a key is not set in `config/settings.json`, the code falls back to the matching
environment variable (`OPENAI_API_KEY` / `ELEVENLABS_API_KEY`).
