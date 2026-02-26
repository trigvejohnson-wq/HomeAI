import queue
import threading
import whisper
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
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
vad_model = load_silero_vad()

# --- VAD-based stop: used so we don't wait a fixed 5s; stop when user pauses ---

def make_vad_stop_callback(
    vad_model,
    silence_duration_ms=700,
    min_recording_sec=0.5,
    tail_sec=2.0,
):
    """
    Returns a callback should_stop(audio, sample_rate) for use with record_until_silence.
    Stops when the last `silence_duration_ms` of audio has no speech (VAD), so recording
    ends shortly after the user stops talking instead of after a fixed timer.
    """
    silence_sec = silence_duration_ms / 1000.0

    def should_stop(audio, sample_rate):
        n = len(audio)
        min_samples = int(min_recording_sec * sample_rate)
        if n < min_samples:
            return False
        # Run VAD on the last `tail_sec` of audio to see where speech ends
        tail_len = min(n, int(tail_sec * sample_rate))
        tail = audio[-tail_len:]
        speech_timestamps = get_speech_timestamps(
            tail,
            vad_model,
            sampling_rate=sample_rate,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )
        if not speech_timestamps:
            # No speech in the tail -> silence; stop
            return True
        last_end = speech_timestamps[-1]["end"]
        silence_at_end_sec = (len(tail) - last_end) / sample_rate
        return silence_at_end_sec >= silence_sec

    return should_stop


def transcribe_audio(audio_data, sample_rate=16000, use_vad=True):

    audio = np.asarray(audio_data, dtype=np.float32)

    if use_vad:
        speech_timestamps = get_speech_timestamps(
            audio,
            vad_model,
            sampling_rate=sample_rate,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )
        if not speech_timestamps:
            return ""
        chunks = [audio[t["start"]:t["end"]] for t in speech_timestamps]
        audio = np.concatenate(chunks)

    if len(audio) == 0:
        return ""

    result = model.transcribe(
        audio,
        fp16=False,
        language="en"
    )
    # Return only the speech text for LLM consumption; never None or extra fields
    return (result.get("text") or "").strip()


# --- Overlapping loop: record next utterance while transcribing the previous ---

def _recorder_thread(audio_queue, sample_rate, vad_stop_cb, max_duration_sec, stop_event):
    """Record utterances with VAD stop and put (audio, sr) on queue. Puts None when done."""
    try:
        while not stop_event.is_set():
            recording, sr = voice_recorder.record_until_silence(
                sample_rate=sample_rate,
                should_stop_cb=vad_stop_cb,
                max_duration_sec=max_duration_sec,
                block_duration_sec=0.3,
                poll_interval_ms=100,
                stop_event=stop_event,
            )
            if stop_event.is_set():
                break
            audio_queue.put((recording, sr))
    finally:
        audio_queue.put(None)


def _transcription_worker(audio_queue, stop_event):
    """Take (audio, sr) from queue and transcribe; print result. Stops on None."""
    while not stop_event.is_set():
        try:
            item = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break
        recording, sr = item
        if len(recording) == 0:
            continue
        text = transcribe_audio(recording, sr, use_vad=True)
        if text:
            print(text)
        print()


if __name__ == "__main__":
    sample_rate = 16000
    max_duration_sec = 15.0
    vad_stop_cb = make_vad_stop_callback(
        vad_model,
        silence_duration_ms=700,
        min_recording_sec=0.5,
        tail_sec=2.0,
    )
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    print("Transcription loop (VAD stop + overlapping). Press Ctrl+C to stop.\n")

    recorder = threading.Thread(
        target=_recorder_thread,
        args=(audio_queue, sample_rate, vad_stop_cb, max_duration_sec, stop_event),
        daemon=True,
    )
    worker = threading.Thread(
        target=_transcription_worker,
        args=(audio_queue, stop_event),
        daemon=True,
    )
    recorder.start()
    worker.start()

    try:
        recorder.join()
        worker.join()
    except KeyboardInterrupt:
        stop_event.set()
        recorder.join()
        worker.join()
        print("\nStopped.")
