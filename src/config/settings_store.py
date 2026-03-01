from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS = {
    "openai_api_key": "",
    "elevenlabs_api_key": "",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "config"


def settings_path() -> Path:
    return config_dir() / "settings.json"


def example_settings_path() -> Path:
    return config_dir() / "settings.example.json"


def _normalize_settings(raw: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if isinstance(raw, dict):
        normalized.update(raw)

    for key, default_value in DEFAULT_SETTINGS.items():
        value = normalized.get(key, default_value)
        normalized[key] = value if isinstance(value, str) else str(value or "")

    return normalized


def ensure_settings_file() -> Path:
    config_dir().mkdir(parents=True, exist_ok=True)
    target = settings_path()
    if target.exists():
        return target

    source = example_settings_path()
    if source.exists():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return target

    save_settings(DEFAULT_SETTINGS)
    return target


def load_settings() -> dict[str, Any]:
    target = ensure_settings_file()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        raw = {}
    settings = _normalize_settings(raw)
    return settings


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_settings(settings)
    config_dir().mkdir(parents=True, exist_ok=True)
    settings_path().write_text(
        json.dumps(normalized, indent=2) + "\n",
        encoding="utf-8",
    )
    return normalized


def get_openai_api_key() -> str:
    value = load_settings().get("openai_api_key", "").strip()
    return value or os.getenv("OPENAI_API_KEY", "")


def get_elevenlabs_api_key() -> str:
    value = load_settings().get("elevenlabs_api_key", "").strip()
    return value or os.getenv("ELEVENLABS_API_KEY", "")
