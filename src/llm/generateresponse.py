import sys
from pathlib import Path

# Ensure project root and src are on path so "config" and "audio" resolve
_root = Path(__file__).resolve().parent.parent.parent
_src = _root / "src"
for _p in (_root, _src):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from openai import OpenAI
from audio.transcribe_audio import transcribe_audio
from audio.voice_recorder import record_audio
from config.settings import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

user_text = transcribe_audio()
if not user_text:
    print("No speech detected. Please run again and speak clearly after \"Listening...\"")
    sys.exit(1)

def response():
  response = client.responses.create(
  model="gpt-4.1-mini",
  input=[
    {
      "role": "system",
      "content": [
        {
          "type": "input_text",
          "text": "# Role and Objective\n- Emulate Luffy from One Piece, using his mannerisms and speech style.\n\n# Instructions\n- Respond accurately, always reflecting Luffy’s core traits and speech patterns.\n- Ensure every reply is consistent and authentic to Luffy.\n\n# Verbosity\n- Keep responses lively and expressive, matching Luffy’s character.\n\n# Stop Conditions\n- Continue until all user queries are answered convincingly in Luffy’s voice."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": user_text
        }
      ]
    }
  ],
  text={
    "format": {
      "type": "text"
    }
  },
  reasoning={},
  tools=[],
  temperature=1,
  max_output_tokens=2048,
  top_p=1,
  store=True,
  include=["web_search_call.action.sources"]
)

if __name__ == "__main__":
    print(response.output_text)