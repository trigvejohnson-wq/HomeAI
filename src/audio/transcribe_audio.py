import argparse
from pathlib import Path

import soundfile as sf
import whisper

try:
    from . import voice_recorder
except ImportError:
    import voice_recorder

# Minimum duration (seconds) to consider we have real speech before transcribing
MIN_SPEECH_DURATION_SEC = 0.5


def _get_device():
    """Prefer CUDA when available, otherwise fall back to CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "cuda"
    except Exception:
        pass
    return "cpu", "cpu"


_device, _device_name = _get_device()
model = whisper.load_model("base", device=_device)


def _transcribe_input(audio_input):
    """Transcribe numpy audio data or an audio file path."""
    result = model.transcribe(audio_input, fp16=_device == "cuda")
    return (result.get("text") or "").strip()


def transcribe_audio(
    silence_duration_sec=1.2,
    min_total_duration_sec=1.5,
    save_recording_path=None,
):
    """Capture speech from microphone until silence and return transcription text."""
    print("Listening... speak now. (Pause ~1 second when done.)")
    should_stop = voice_recorder.make_silence_stop_callback(
        silence_duration_sec=silence_duration_sec,
        min_total_duration_sec=min_total_duration_sec,
    )
    audio_data, sample_rate = voice_recorder.record_until_silence(
        should_stop_cb=should_stop,
    )

    if save_recording_path:
        target = Path(save_recording_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target), audio_data, sample_rate)
        print(f"Saved microphone capture to: {target}")

    duration_sec = len(audio_data) / sample_rate if len(audio_data) else 0
    if duration_sec < MIN_SPEECH_DURATION_SEC:
        return ""  # Caller should treat as "no speech captured"

    # Whisper expects 16 kHz mono float32; record_until_silence uses 16 kHz by default.
    return _transcribe_input(audio_data)


def transcribe_audio_file(audio_file_path):
    """Transcribe an existing audio file."""
    audio_path = Path(audio_file_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return _transcribe_input(str(audio_path))


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Transcribe voice input from microphone or an audio file."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to an existing audio file (wav/mp3/m4a/etc.) to transcribe.",
    )
    parser.add_argument(
        "--save-recording",
        type=str,
        help="If using microphone mode, save recorded audio to this file path.",
    )
    parser.add_argument(
        "--silence-duration-sec",
        type=float,
        default=1.2,
        help="Continuous silence duration required to stop recording.",
    )
    parser.add_argument(
        "--min-total-duration-sec",
        type=float,
        default=1.5,
        help="Minimum microphone capture length before silence-stop is allowed.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    print(f"Whisper device: {_device_name}")
    if args.input_file:
        print(transcribe_audio_file(args.input_file))
    else:
        print(
            transcribe_audio(
                silence_duration_sec=args.silence_duration_sec,
                min_total_duration_sec=args.min_total_duration_sec,
                save_recording_path=args.save_recording,
            )
        )
