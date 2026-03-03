THIS PROJECT IS CURRENTLY UNFISHINED!

The idea of this project is to allow the user to set up a Local AI Home Assistant. User's will be given the option to use a third party service for custom voice TTS.
You will be able to also set the personality of the Home AI Assistant, and it will randomly make comments based on the environment of your house.
You will also be able to ask key questions on where a person/pet is within the house, along with any key objects that you may be looking for.

------------------------------------------------------------------------------------------------------------------------------------------------

# AI Home Assistant — Project Completion Status

This document summarizes how much of the project is **done** versus **left to do**, based on the goals in the README and the architecture in the AI Custom Voice Personality Guide.

---

## Project Goals (from README)

1. **Local AI Home Assistant** — voice-driven, with optional custom TTS.
2. **Configurable personality** — set the assistant’s character; random comments based on home environment.
3. **Where is person/pet?** — ask where someone or a pet is in the house.
4. **Key objects** — ask about or search for specific objects.

---

## Completed (~40%)

| Area | Status | Notes |
|------|--------|------|
| **Microphone & speech-to-text** | Done | `voice_recorder.py`: record until silence (VAD-style). `transcribe_audio.py`: Whisper (local, GPU/CPU). |
| **LLM integration** | Done (single turn) | `generateresponse.py`: records → transcribes → calls OpenAI Responses API → returns text. Personality is **hardcoded** (Luffy). |
| **Vision basics** | Done | `vision.py`: capture frame from camera, encode frame to base64. Not yet wired into LLM or “where is X” logic. |
| **TTS (Edge)** | Partial | `edgy.py`: async Edge TTS function exists. Not called from pipeline; `if __name__` block has bugs (missing `asyncio.run`, wrong `response` reference). |
| **Config** | Minimal | `config/settings.py`: loads `.env`; only `OPENAI_API_KEY` used. `.env.example` present. |
| **Documentation** | Done | `docs/AI_Custom_Voice_Personality_Guide.md`: full pipeline (STT, vision, LLM, TTS, RVC, orchestration, config). |

**Rough completion:** Core input (mic + Whisper) and one-shot “speak → LLM → text” path work. Vision and TTS exist as building blocks but are not part of a single runnable assistant.

---

## Remaining (~60%)

| Area | What’s left |
|------|-------------|
| **Orchestration** | No main entry point. Implement something like `run_one_turn()` (record → transcribe → optional camera → LLM → TTS → play) and a `main_loop()` for continuous conversation. |
| **Personality** | Move from hardcoded Luffy to config (e.g. `config.yaml` or `.env`) so users can set name, traits, and system prompt. |
| **Environment-based comments** | Use camera (and optionally other sensors) so the assistant can make “random” or contextual comments about the environment; needs vision → LLM integration and a way to trigger (schedule or event). |
| **“Where is person/pet?”** | Add vision analysis (e.g. YOLO for people/pets/objects), build scene descriptions or structured detections, and feed that into the LLM so it can answer “where is X?” and “where is [object]?”. |
| **TTS in pipeline** | Fix `edgy.py` and call it from the main flow; play generated audio (e.g. with `sounddevice` or `scipy.io.wavfile`). |
| **Custom voice (third-party)** | Optional: add RVC (or other) voice conversion as in the guide (base TTS → RVC → play); make it configurable (paths, pitch, etc.). |
| **Configuration** | Extend beyond API key: personality, TTS provider/voice, RVC paths, camera index, record duration, LLM model, etc. (e.g. `config.yaml` as in the guide). |
| **Real-time loop** | Optional: wake word, push-to-talk, or continuous listen so the assistant runs as a persistent home assistant, not just a one-off script. |

---

## Summary

- **Done:** Input pipeline (mic + Whisper), single-turn LLM with fixed personality, basic vision capture/encoding, and a partial TTS module. Config is minimal; docs are in place.
- **Left:** End-to-end orchestration (one turn + loop), configurable personality, environment-based behavior, “where is person/pet/object” using vision, TTS wired and played, optional custom voice and full config.

**Overall: ~40% complete; ~60% remaining** (by scope of features, not by lines of code).
