"""Thin client for the StackChan Hermes gateway (http://127.0.0.1:8767). Every call is fail-soft:
when StackChan is offline the game keeps running and the phone page shows what it would have said."""

from __future__ import annotations

import json
import logging
import threading
import urllib.request

log = logging.getLogger("stackchan")
GATEWAY = "http://127.0.0.1:8767"


def _call(method, path, body=None, timeout=10.0):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(GATEWAY + path, data=data, method=method,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except Exception as e:  # offline, timeout, device error
        log.info("stackchan %s %s failed: %s", method, path, e)
        return None


class StackChan:
    def __init__(self):
        self.last_said = ""
        self._speech_lock = threading.Lock()

    def online(self) -> bool:
        h = _call("GET", "/health", timeout=2)
        return bool(h and h.get("deviceOnline"))

    def say(self, text: str, wait: bool = False, expression: str | None = None):
        """Speak (Irodori TTS on the Mac -> StackChan speaker). Non-blocking unless wait=True."""
        self.last_said = text
        log.info("SAY: %s", text)

        def run():
            with self._speech_lock:
                if expression:
                    _call("POST", "/api/expression", {"expression": expression})
                _call("POST", "/api/speak", {"text": text}, timeout=90)

        if wait:
            run()
        else:
            threading.Thread(target=run, daemon=True).start()

    def expression(self, name: str):
        _call("POST", "/api/expression", {"expression": name})

    def look(self, yaw: int, pitch: int, speed: int = 150):
        _call("POST", "/api/look", {"yaw": yaw, "pitch": pitch, "speed": speed})

    def show_qr(self, url: str, caption: str = ""):
        return _call("POST", "/api/qr", {"text": url, "caption": caption})

    def hide_qr(self):
        return _call("POST", "/api/qr/hide", {})

    def text(self, text: str):
        _call("POST", "/api/display-text", {"text": text})

    def led(self, mode: str, rgb=(0, 0, 0), ms: int = 0):
        """LED ring: off|solid|pulse|blink|rainbow|chase (non-blocking)."""
        r, g, b = rgb
        threading.Thread(target=_call, args=("POST", "/api/led-effect",
                         {"mode": mode, "red": r, "green": g, "blue": b, "ms": ms}, 3), daemon=True).start()

    def dance(self, ms: int = 4000):
        threading.Thread(target=_call, args=("POST", "/api/dance", {"ms": ms}, 3), daemon=True).start()

    def touch_log(self, after: int):
        return _call("GET", f"/api/touch-log?after={after}", timeout=3) or {"last": after, "events": []}
