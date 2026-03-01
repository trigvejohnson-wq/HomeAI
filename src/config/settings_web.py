from __future__ import annotations

import argparse
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from settings_store import load_settings, save_settings


class SettingsHandler(BaseHTTPRequestHandler):
    server_version = "LocalAssistantSettings/1.0"

    def _send_html(self, content: str, status: int = HTTPStatus.OK) -> None:
        content_bytes = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content_bytes)))
        self.end_headers()
        self.wfile.write(content_bytes)

    def _render_page(self, success: bool = False) -> str:
        settings = load_settings()
        openai_key = html.escape(settings.get("openai_api_key", ""))
        elevenlabs_key = html.escape(settings.get("elevenlabs_api_key", ""))

        success_banner = (
            "<div class='notice success'>Settings saved.</div>" if success else ""
        )

        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Local AI Assistant Settings</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg-deep: #080510;
        --bg-mid: #120a24;
        --panel: #1a1033cc;
        --panel-border: #7d5bb04d;
        --text-primary: #f4ecff;
        --text-secondary: #c8b7e6;
        --accent: #a66cff;
        --accent-strong: #874ef2;
        --accent-soft: #b997ff33;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        padding: 24px;
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        color: var(--text-primary);
        background:
          radial-gradient(900px 500px at 15% 10%, #4a208433 0%, transparent 55%),
          radial-gradient(700px 500px at 85% 90%, #8434d933 0%, transparent 60%),
          linear-gradient(160deg, var(--bg-deep), var(--bg-mid));
      }}

      .card {{
        width: min(720px, 100%);
        border-radius: 18px;
        border: 1px solid var(--panel-border);
        background: linear-gradient(180deg, #251248b3 0%, #140a28e6 100%);
        box-shadow:
          0 30px 80px #05030d99,
          inset 0 1px 0 #ffffff14;
        backdrop-filter: blur(6px);
        padding: 28px;
      }}

      .eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #dcc8ff;
        background: #7e4de633;
        border: 1px solid #b997ff66;
        padding: 6px 10px;
        border-radius: 999px;
        margin-bottom: 16px;
      }}

      h1 {{
        margin: 0 0 10px;
        font-size: clamp(1.5rem, 2.3vw, 1.9rem);
        line-height: 1.2;
      }}

      .subtitle {{
        margin: 0 0 18px;
        color: var(--text-secondary);
        line-height: 1.5;
      }}

      .notice {{
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 18px;
        font-weight: 600;
        border: 1px solid #8f63e466;
        background: #7c56cf26;
      }}

      .success {{
        border-color: #a882f266;
        background: #9468eb33;
      }}

      form {{
        display: grid;
        gap: 14px;
      }}

      .field {{
        display: grid;
        gap: 8px;
      }}

      label {{
        font-size: 0.88rem;
        color: #ddd0f5;
      }}

      input[type="password"] {{
        width: 100%;
        border: 1px solid #8964c680;
        border-radius: 12px;
        background: #0d071b;
        color: var(--text-primary);
        padding: 12px 13px;
        font-size: 0.95rem;
        transition: border-color 120ms ease, box-shadow 120ms ease;
      }}

      input[type="password"]:focus {{
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 4px var(--accent-soft);
      }}

      button {{
        margin-top: 4px;
        border: 1px solid #b38dff66;
        border-radius: 12px;
        padding: 11px 14px;
        color: #fff;
        font-weight: 700;
        font-size: 0.95rem;
        cursor: pointer;
        background: linear-gradient(180deg, var(--accent) 0%, var(--accent-strong) 100%);
        box-shadow: 0 10px 30px #5e33ad66;
        transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
      }}

      button:hover {{
        filter: brightness(1.06);
        box-shadow: 0 14px 34px #5e33ad80;
      }}

      button:active {{
        transform: translateY(1px);
      }}

      .hint {{
        margin-top: 2px;
        font-size: 0.84rem;
        color: #b6a3d6;
      }}

      code {{
        color: #f5ebff;
        background: #5931993d;
        border: 1px solid #b58dff33;
        border-radius: 6px;
        padding: 1px 6px;
        font-size: 0.84rem;
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <div class="eyebrow">Local only</div>
      <h1>Local AI Assistant Settings</h1>
      <p class="subtitle">Manage API credentials for this project. Values are stored in <code>config/settings.json</code>.</p>
      {success_banner}
      <form method="post" action="/save">
        <div class="field">
          <label for="openai_api_key">OpenAI API key</label>
          <input id="openai_api_key" name="openai_api_key" type="password" value="{openai_key}" autocomplete="off" />
        </div>
        <div class="field">
          <label for="elevenlabs_api_key">ElevenLabs API key</label>
          <input id="elevenlabs_api_key" name="elevenlabs_api_key" type="password" value="{elevenlabs_key}" autocomplete="off" />
        </div>
        <button type="submit">Save settings</button>
        <div class="hint">Leave a field empty if you want to remove that API key.</div>
      </form>
    </div>
  </body>
</html>
"""

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if parsed_url.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        query = parse_qs(parsed_url.query)
        was_saved = query.get("saved", ["0"])[0] == "1"
        self._send_html(self._render_page(success=was_saved))

    def do_POST(self) -> None:
        if self.path != "/save":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        body = self.rfile.read(content_length).decode("utf-8")
        form_data = parse_qs(body, keep_blank_values=True)
        save_settings(
            {
                "openai_api_key": form_data.get("openai_api_key", [""])[0].strip(),
                "elevenlabs_api_key": form_data.get("elevenlabs_api_key", [""])[0].strip(),
            }
        )

        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/?saved=1")
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local web UI to manage API key configuration."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind (default: 8765).",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SettingsHandler)
    print(f"Settings UI running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
