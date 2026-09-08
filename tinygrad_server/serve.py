#!/usr/bin/env python3
"""
tinygrad-backed TTS server exposing /v1/audio/speech, the contract AllTalk's
VibeVoice and Qwen3-TTS engines speak when their "backend_type" setting is
"network" (see system/tts_engines/vibevoice/model_engine.py and
system/tts_engines/qwen3tts/model_engine.py, and the "Remote server contract"
accordion on each engine's settings page).

This is the transport layer only. Model backends (backends/vibevoice.py,
backends/qwen3_tts.py) are stubbed and return 501 until the actual tinygrad
model ports land -- see the docstrings in those files for what's missing.

Run: python serve.py --port 8100 [--api-key SECRET]
"""
import argparse
import json
import socketserver
import sys
import time
from http.server import BaseHTTPRequestHandler
from typing import Optional

from backends import vibevoice, qwen3_tts

BACKENDS = {"vibevoice": vibevoice, "qwen3_tts": qwen3_tts}


def _log(msg: str) -> None:
    sys.stderr.write(msg)
    sys.stderr.flush()


def _backend_for(model_id: str):
    s = (model_id or "").lower()
    if "vibevoice" in s:
        return BACKENDS["vibevoice"]
    if "qwen3-tts" in s or "qwen3_tts" in s:
        return BACKENDS["qwen3_tts"]
    raise ValueError(f"no backend registered for model '{model_id}'")


def _all_model_ids():
    return vibevoice.MODEL_IDS + qwen3_tts.MODEL_IDS


class Handler(BaseHTTPRequestHandler):
    server: "TTSServer"

    def log_message(self, fmt, *args):
        pass  # quiet; we log ourselves below with more useful detail

    def _send(self, data: bytes, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json_error(self, message: str, status_code: int):
        self._send(json.dumps({"error": message}).encode("utf-8"), status_code=status_code)

    def _authorized(self) -> bool:
        if not self.server.api_key:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.server.api_key}"

    def do_GET(self):
        if self.path == "/v1/models":
            data = json.dumps({"object": "list", "data": [{"id": m, "object": "model"} for m in _all_model_ids()]})
            return self._send(data.encode("utf-8"))
        if self.path == "/health":
            return self._send(b'{"status":"ok"}')
        return self._send(b"tinygrad TTS server. See /v1/models or POST /v1/audio/speech.\n", content_type="text/plain")

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            return self._send_json_error(f"unknown path '{self.path}'", 404)
        if not self._authorized():
            return self._send_json_error("unauthorized", 401)

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as e:
            return self._send_json_error(f"invalid JSON body: {e}", 400)

        model_id = payload.get("model", "")
        st = time.perf_counter()
        _log(f"POST /v1/audio/speech  model={model_id!r} mode={payload.get('mode', '-')}\n")

        try:
            backend = _backend_for(model_id)
        except ValueError as e:
            return self._send_json_error(str(e), 400)

        try:
            audio_bytes, content_type = backend.generate(payload)
        except NotImplementedError as e:
            return self._send_json_error(str(e), 501)
        except ValueError as e:
            return self._send_json_error(str(e), 400)
        except Exception as e:  # defensive: mirrors the AllTalk engines' own generation error handling
            _log(f"generation failed: {e}\n")
            return self._send_json_error(str(e), 500)

        _log(f"done in {time.perf_counter() - st:.2f}s\n")
        return self._send(audio_bytes, content_type=content_type)


class TTSServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, api_key: Optional[str] = None):
        self.api_key = api_key
        super().__init__(server_address, Handler)


def main():
    parser = argparse.ArgumentParser(description="tinygrad TTS server (/v1/audio/speech)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--api-key", default=None, help="If set, requests must send 'Authorization: Bearer <key>'")
    args = parser.parse_args()

    server = TTSServer((args.host, args.port), api_key=args.api_key)
    print(f"tinygrad TTS server listening on http://{args.host}:{args.port}")
    print(f"Models: {', '.join(_all_model_ids())}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
