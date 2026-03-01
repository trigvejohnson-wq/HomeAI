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
        color-scheme: light dark;
      }}
      body {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        max-width: 680px;
        margin: 40px auto;
        padding: 0 20px;
      }}
      .card {{
        border: 1px solid #8884;
        border-radius: 12px;
        padding: 24px;
      }}
      h1 {{
        margin-top: 0;
      }}
      .field {{
        margin-bottom: 16px;
      }}
      label {{
        display: block;
        margin-bottom: 8px;
        font-weight: 600;
      }}
      input[type="password"] {{
        width: 100%;
        box-sizing: border-box;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #8885;
      }}
      button {{
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #8885;
        cursor: pointer;
      }}
      .notice {{
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 16px;
      }}
      .success {{
        border: 1px solid #2b8a3e66;
        background: #2b8a3e1f;
      }}
      .hint {{
        margin-top: 4px;
        font-size: 0.9rem;
        color: #666;
      }}
      code {{
        font-size: 0.92rem;
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Local AI Assistant Settings</h1>
      <p>Update API credentials used by this project. Values are stored in <code>config/settings.json</code>.</p>
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
