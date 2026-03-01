from elevenlabs import ElevenLabs
from pathlib import Path
import sys

_CONFIG_MODULE_DIR = Path(__file__).resolve().parents[1] / "config"
if str(_CONFIG_MODULE_DIR) not in sys.path:
    sys.path.append(str(_CONFIG_MODULE_DIR))

from settings_store import get_elevenlabs_api_key

_api_key = get_elevenlabs_api_key()
client = ElevenLabs(api_key=_api_key) if _api_key else ElevenLabs()

def speak_with_elevenlabs(text, voice_id="your-voice-id"):
    audio = client.text_to_speech.convert(voice_id, text=text)
    return audio  # bytes