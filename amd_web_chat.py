#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Browser chat for the YASAKANI AMD/HIP fork on Linux (no GUI).

Reuses the upstream quickstart web proxy unchanged: quickstart.make_web_handler + quickstart.LocalServer
serve quickstart/chat.html on 127.0.0.1 and relay /health, /v1/models and /v1/chat/completions (SSE) to
`yasakani serve`.  This helper only starts the server, waits until it is ready, opens the browser and
shuts both down cleanly.  The proxy logic itself is NOT reimplemented here.

Usage: python3 amd_web_chat.py --model-dir DIR [--dll PATH] [--no-browser] [--web-port N]
                               [--engine-port N] [--lang en|ja] [yasakani serve options...]
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "quickstart"))
# The bundled package works without installation; the serve subprocess inherits this.
if (ROOT / "python" / "yasakani" / "__init__.py").is_file():
    os.environ["PYTHONPATH"] = str(ROOT / "python") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

import quickstart as qs  # noqa: E402  (stdlib-only module; its main() is guarded and Windows-gated, the import is safe on Linux)


def _fail(msg: str) -> int:
    print("error: " + msg, file=sys.stderr, flush=True)
    return 2


def _opt_value(rest, name):
    """The value of --name V or --name=V inside the pass-through options (None when absent)."""
    for i, t in enumerate(rest):
        if t == name and i + 1 < len(rest):
            return rest[i + 1]
        if t.startswith(name + "="):
            return t.split("=", 1)[1]
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="YASAKANI browser chat (upstream quickstart proxy + yasakani serve)", allow_abbrev=False)
    ap.add_argument("--model-dir", required=True, help="model folder with yasakani.json")
    ap.add_argument("--dll", help="path of libyskn.so (default: $YASAKANI_DLL or next to the package)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window (the URL is printed)")
    ap.add_argument("--engine-port", type=int, help="port of yasakani serve (default: 8765 or the next free one; --port wins)")
    ap.add_argument("--web-port", type=int, help="port of the chat page (default: 8766 or the next free one)")
    ap.add_argument("--lang", choices=("en", "ja"), default="en", help="language of the chat page (default en)")
    a, rest = ap.parse_known_args(argv)

    host = _opt_value(rest, "--host")
    if host and host not in ("127.0.0.1", "localhost", "::1"):
        return _fail("the browser chat listens on 127.0.0.1 only; use --mode api for --host " + host)
    if _opt_value(rest, "--api-key") is not None or os.environ.get("YASAKANI_API_KEY"):
        return _fail("the browser chat cannot send an API key; use --mode api with --api-key")

    page_path = ROOT / "quickstart" / "chat.html"
    if not page_path.is_file():
        return _fail("chat.html not found: %s (unpack the distribution again)" % page_path)
    page = page_path.read_bytes()

    given_port = _opt_value(rest, "--port")
    eport = int(given_port or a.engine_port or qs.pick_port(qs.ENGINE_PORT))
    wport = a.web_port or qs.pick_port(qs.WEB_PORT, avoid=(eport,))

    cmd = [sys.executable, "-m", "yasakani", "serve", "--model-dir", a.model_dir]
    if a.dll:
        cmd += ["--dll", a.dll]
    cmd += rest
    if given_port is None:
        cmd += ["--port", str(eport)]
    print("> " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)

    # Ctrl+C and SIGTERM (systemctl stop) both take the orderly exit below.
    def _interrupt(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _interrupt)

    httpd = None
    try:
        if not qs.wait_ready(eport, proc):
            rc = proc.poll()
            return _fail("yasakani serve did not become ready%s. See its messages above "
                         "(common causes: wrong model folder, missing ROCm runtime, port in use)."
                         % (" (exit code %s)" % rc if rc is not None else ""))
        conf = {"lang": a.lang, "model": "custom", "version": qs.QS_VERSION, "repo": qs.REPO_URL, "support": qs.SUPPORT_URL}
        try:
            httpd = qs.LocalServer(("127.0.0.1", wport), qs.make_web_handler(page, conf, eport, wport))
        except OSError as ex:
            return _fail("cannot listen on 127.0.0.1:%d (%s). Use --web-port to pick another port." % (wport, ex))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = "http://127.0.0.1:%d/" % wport
        print("\nChat in your browser: %s\n(The OpenAI-compatible API is at http://127.0.0.1:%d/v1 . Ctrl+C stops both.)"
              % (url, eport), flush=True)
        if not a.no_browser:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - the address is printed above
                pass
        proc.wait()
        return proc.returncode or 0          # the server ended on its own: pass its exit code on
    except KeyboardInterrupt:
        return 0
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)        # Ctrl+C reached serve too: let it close the engine first
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("browser chat stopped.", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
