import whisper

try:
    from . import voice_recorder
except ImportError:
    import voice_recorder

# Prefer NVIDIA GPU (CUDA), then CPU
def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "cuda"
    except Exception:
        pass
    return "cpu", "cpu"

_device, _device_name = _get_device()
model = whisper.load_model("base", device=_device)
#print(f"Whisper using device: {_device_name}")

# Minimum duration (seconds) to consider we have real speech before transcribing
MIN_SPEECH_DURATION_SEC = 0.5

def transcribe_audio():
    print("Listening... speak now. (Pause ~1 second when done.)")
    should_stop = voice_recorder.make_silence_stop_callback(
        silence_duration_sec=1.2,
        min_total_duration_sec=1.5,
    )
    audio_data, sample_rate = voice_recorder.record_until_silence(
        should_stop_cb=should_stop,
    )
    duration_sec = len(audio_data) / sample_rate if len(audio_data) else 0
    if duration_sec < MIN_SPEECH_DURATION_SEC:
        return ""  # Caller should treat as "no speech captured"
    # Whisper expects 16 kHz mono float32; record_until_silence uses 16 kHz by default
    result = model.transcribe(audio_data)
    text = (result.get("text") or "").strip()
    return text


if __name__ == "__main__":
    print(transcribe_audio())