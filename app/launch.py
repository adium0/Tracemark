"""Start the demo server on http://127.0.0.1:8000 and open it in the browser.

Called by run.sh / run.bat as:  python -m app.launch
Environment variables:
  PORT=8001        use another port
  NO_BROWSER=1     don't open a browser window
"""
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

HOST = "127.0.0.1"


def _port_busy(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, port)) == 0


def _health(port, timeout=2.0):
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/health", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _open_browser(url):
    if os.environ.get("NO_BROWSER"):
        return
    try:
        opened = webbrowser.open(url, new=2)
    except Exception:
        opened = False
    if not opened:
        print(f"\n  Couldn't open a browser automatically. Open this address yourself: {url}\n", flush=True)


def _open_when_ready(port, url):
    deadline = time.time() + 180
    while time.time() < deadline:
        info = _health(port, timeout=1.0)
        if info and info.get("status") == "ok":
            print(f"\n  Integrity Theory demo is ready: {url}\n"
                  "  Keep this window open while you use it. Press Ctrl+C to stop.\n", flush=True)
            _open_browser(url)
            return
        time.sleep(0.5)
    print(f"\n  The server is taking unusually long to start. Try opening {url} yourself.\n", flush=True)


def main():
    port = int(os.environ.get("PORT", "8000"))
    url = f"http://{HOST}:{port}/"

    if _port_busy(port):
        info = _health(port)
        if info and info.get("app") == "integrity-demo":
            print(f"  The demo is already running at {url} - opening it.")
            _open_browser(url)
            return 0
        print(f"\n  ERROR: port {port} is already used by another program.\n"
              f"  Close that program, or start the demo on another port:\n"
              f"    Mac/Linux:  PORT=8001 ./run.sh\n"
              f"    Windows:    set PORT=8001 && run.bat\n", file=sys.stderr)
        return 1

    import uvicorn  # imported here so the checks above stay fast

    threading.Thread(target=_open_when_ready, args=(port, url), daemon=True).start()
    print(f"  Loading the model and starting the server on {url} ...", flush=True)
    uvicorn.run("app.server:app", host=HOST, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
