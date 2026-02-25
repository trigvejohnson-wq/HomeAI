from elevenlabs import ElevenLabs
import os

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def speak_with_elevenlabs(text, voice_id="your-voice-id"):
    audio = client.text_to_speech.convert(voice_id, text=text)
    return audio  # bytes