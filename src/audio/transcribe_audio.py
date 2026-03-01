import whisper
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
print(f"Whisper using device: {_device_name}")

def transcribe_audio():
    should_stop = voice_recorder.make_silence_stop_callback()
    audio_data, _ = voice_recorder.record_until_silence(
        should_stop_cb=should_stop,
    )
    # Whisper expects 16 kHz mono float32; record_until_silence uses 16 kHz by default
    result = model.transcribe(audio_data)
    return result["text"]


if __name__ == "__main__":
    print(transcribe_audio())