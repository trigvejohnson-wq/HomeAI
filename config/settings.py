import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (parent of config/)
_load_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
