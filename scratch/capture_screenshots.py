"""
capture_screenshots.py - Launch Chrome with remote debugging and capture
high-resolution screenshots of all 6 pages of the Streamlit dashboard via CDP.
"""

import subprocess
import time
import json
import base64
import os
import urllib.request
import websocket

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
USER_DATA_DIR = os.path.abspath("outputs/dashboard_screenshots/chrome_profile")
PORT = 9222
BASE_URL = "http://localhost:8501"

PAGES = [
    ("00_start_here.png", f"{BASE_URL}/?page=0"),
    ("01_fleet_overview.png", f"{BASE_URL}/?page=1"),
    ("02_engine_detail.png", f"{BASE_URL}/?page=2"),
    ("03_decision_log.png", f"{BASE_URL}/?page=3"),
    ("04_policy_comparison.png", f"{BASE_URL}/?page=4"),
    ("05_sensitivity_limitations.png", f"{BASE_URL}/?page=5"),
    ("06_technical_architecture.png", f"{BASE_URL}/?page=6"),
    ("07_live_decision_demo.png", f"{BASE_URL}/?page=7"),
    ("08_langgraph_agents.png", f"{BASE_URL}/?page=8&llm=groq"),
]


def send_cdp_command(ws, method, params=None, msg_id=1):
    msg = {"id": msg_id, "method": method}
    if params:
        msg["params"] = params
    ws.send(json.dumps(msg))
    while True:
        resp = json.loads(ws.recv())
        if resp.get("id") == msg_id:
            return resp.get("result", {})


def capture_all():
    os.makedirs("outputs/dashboard_screenshots", exist_ok=True)

    # Launch Chrome headless with remote debugging
    chrome_proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--remote-allow-origins=*",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={USER_DATA_DIR}",
        "--window-size=1440,900",
    ])

    try:
        # Wait for debugging port to become active
        print("Waiting for Chrome to initialize CDP on port", PORT)
        for _ in range(15):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}/json/version", timeout=1) as resp:
                    if resp.status == 200:
                        print("Chrome CDP ready!")
                        break
            except Exception:
                pass
        else:
            raise RuntimeError("Chrome failed to start CDP")

        pages_to_capture = PAGES
        import sys
        if len(sys.argv) > 1:
            target = sys.argv[1].lower()
            pages_to_capture = [p for p in PAGES if target in p[0].lower() or target in p[1].lower()]

        for filename, url in pages_to_capture:
            out_file = os.path.abspath(os.path.join("outputs/dashboard_screenshots", filename))
            print(f"Navigating to {url} -> {filename}...")

            # Open a new tab
            req = urllib.request.Request(f"http://localhost:{PORT}/json/new?{url}", method="PUT")
            with urllib.request.urlopen(req) as resp:
                tab_data = json.loads(resp.read().decode("utf-8"))

            ws_url = tab_data["webSocketDebuggerUrl"]
            tab_id = tab_data["id"]

            ws = websocket.create_connection(ws_url, timeout=15)
            try:
                # Enable Page domain
                send_cdp_command(ws, "Page.enable", msg_id=1)
                send_cdp_command(ws, "Emulation.setDeviceMetricsOverride", {
                    "width": 1440,
                    "height": 1050,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                }, msg_id=2)

                # Wait for Streamlit to connect and render
                time.sleep(12.0 if ("page=0" in url or "page=1" in url) else (16.0 if "page=8" in url else 6.0))

                # Capture full page screenshot
                res = send_cdp_command(ws, "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}, msg_id=3)
                img_bytes = base64.b64decode(res["data"])

                with open(out_file, "wb") as f:
                    f.write(img_bytes)

                print(f"  Saved {filename}: {len(img_bytes):,} bytes ({out_file})")

            finally:
                ws.close()
                # Close the tab
                try:
                    req_close = urllib.request.Request(f"http://localhost:{PORT}/json/close/{tab_id}", method="PUT")
                    urllib.request.urlopen(req_close, timeout=2)
                except Exception:
                    pass

    finally:
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=3)
        except Exception:
            chrome_proc.kill()


if __name__ == "__main__":
    capture_all()
