import threading
import sounddevice as sd
import numpy as np


def make_silence_stop_callback(
    silence_duration_sec=1.2,
    min_speech_sec=0.3,
    silence_threshold=0.012,
    min_total_duration_sec=1.5,
):
    """
    Returns a should_stop_cb for record_until_silence that stops after continuous
    silence. Uses RMS (root mean square) of recent audio to detect silence.

    Args:
        silence_duration_sec: Stop after this many seconds of continuous silence.
        min_speech_sec: Require at least this much speech before stopping on silence.
        silence_threshold: RMS below this (float32) is considered silence.
        min_total_duration_sec: Don't stop on silence until at least this much
            recording time has passed (gives user time to start speaking).
    """
    last_speech_time = [0.0]

    def should_stop(audio_so_far, sample_rate):
        if len(audio_so_far) == 0:
            return False
        lookback_sec = 0.3
        lookback_samples = int(lookback_sec * sample_rate)
        recent = (
            audio_so_far[-lookback_samples:]
            if len(audio_so_far) >= lookback_samples
            else audio_so_far
        )
        rms = np.sqrt(np.mean(recent**2))
        total_duration = len(audio_so_far) / sample_rate

        # Don't allow stop-on-silence until user has had time to start speaking
        if total_duration < min_total_duration_sec:
            return False
        if rms > silence_threshold:
            last_speech_time[0] = total_duration
            return False
        if total_duration < min_speech_sec:
            return False
        silence_length = total_duration - last_speech_time[0]
        return silence_length >= silence_duration_sec

    return should_stop


def record_audio(duration_sec=5, sample_rate=16000):
    """Record audio from default microphone for a fixed duration."""
    #print("Recording audio for", duration_sec, "seconds...")
    audio = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    return audio.flatten(), sample_rate


def record_until_silence(
    sample_rate=16000,
    should_stop_cb=None,
    max_duration_sec=15.0,
    block_duration_sec=0.75,
    poll_interval_ms=250,
    stop_event=None,
):
    """
    Record in small blocks until should_stop_cb(audio_so_far, sample_rate) returns True,
    or max_duration_sec is reached. Uses a streaming callback so we can stop as soon as
    the user stops speaking (e.g. via VAD), instead of waiting a fixed duration.

    Args:
        sample_rate: Sample rate in Hz.
        should_stop_cb: Callable(audio_so_far: np.ndarray, sample_rate: int) -> bool.
            Called periodically; return True to stop recording and return the buffer.
        max_duration_sec: Stop recording after this many seconds regardless of VAD.
        block_duration_sec: Size of each block from the microphone (smaller = more responsive).
        poll_interval_ms: How often to check should_stop_cb (milliseconds).
        stop_event: Optional threading.Event(); if set, stop recording immediately (e.g. Ctrl+C).

    Returns:
        (audio_float32_array, sample_rate). audio may be empty if stopped immediately.
    """
    if should_stop_cb is None:
        return record_audio(max_duration_sec, sample_rate)

    #print("Recording... (speak, then pause to stop)")
    block_size = int(block_duration_sec * sample_rate)
    max_samples = int(max_duration_sec * sample_rate)
    buffer = []
    buffer_lock = threading.Lock()

    def stream_callback(indata, frames, time_info, status):
        if status:
            print("(stream)", status)
        with buffer_lock:
            buffer.append(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=block_size,
        callback=stream_callback,
    ):
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            sd.sleep(poll_interval_ms)
            with buffer_lock:
                if not buffer:
                    audio_so_far = np.array([], dtype=np.float32)
                else:
                    audio_so_far = np.concatenate(buffer).flatten()
            total_samples = len(audio_so_far)
            if total_samples >= max_samples:
                break
            if total_samples > 0 and should_stop_cb(audio_so_far, sample_rate):
                break

    if not buffer:
        return np.array([], dtype=np.float32), sample_rate
    with buffer_lock:
        return np.concatenate(buffer).flatten(), sample_rate