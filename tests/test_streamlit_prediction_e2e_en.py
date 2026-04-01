import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "eri_q4_streamlit_app_en.py"
STARTUP_TIMEOUT_SECONDS = 60


class StreamlitApp:
    def __init__(self, proc, url):
        self.proc = proc
        self.url = url
        self.stdout = ""
        self.stderr = ""

    @classmethod
    def start(cls):
        port = _find_free_port()
        env = os.environ.copy()
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                APP_FILE.name,
                "--server.headless",
                "true",
                "--server.port",
                str(port),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        app = cls(proc=proc, url=f"http://127.0.0.1:{port}")
        try:
            _wait_for_server(app.url, proc)
        except Exception:
            app.stop()
            raise
        return app

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.proc.stdout or self.proc.stderr:
            try:
                self.stdout, self.stderr = self.proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.stdout, self.stderr = self.proc.communicate()
        return self.stdout, self.stderr


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(url, proc):
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"Streamlit exited before startup.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for Streamlit at {url}")


def _fill_visible_number(page, label, value):
    locator = page.locator(f'input[aria-label="{label}"]:visible')
    locator.fill(str(value))
    actual = locator.input_value()
    assert abs(float(actual) - float(value)) < 1e-9, f"{label} expected {value}, got {actual}"


def _wait_for_visible_text(page, text, timeout_ms=15000):
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        body_text = page.locator("body").inner_text()
        if text in body_text:
            return body_text
        page.wait_for_timeout(250)
    raise AssertionError(f"Timed out waiting for visible text: {text}")


def _assert_result_text(body_text, result_title):
    assert result_title in body_text
    assert "Training-set ROC optimal cutoff" in body_text
    assert "ERI Top 25%" in body_text
    assert re.search(r"\d+\.\d+%\s+probability", body_text, flags=re.IGNORECASE), body_text


def test_streamlit_prediction_flow_english():
    app = StreamlitApp.start()
    page_errors = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1700, "height": 1800})
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            page.goto(app.url, wait_until="networkidle")
            _wait_for_visible_text(page, "Compact Calculator", timeout_ms=20000)
            body = _wait_for_visible_text(page, "Full Calculator", timeout_ms=20000)
            assert "Next-Quarter ESA Resistance Risk Predictor" in body
            assert "Compact Calculator" in body
            assert "Full Calculator" in body

            _fill_visible_number(page, "Dry weight (kg)", 62.5)
            _fill_visible_number(page, "Hemoglobin (g/L)", 118)
            _fill_visible_number(page, "Age (years)", 55)
            _fill_visible_number(page, "spKt/V", 1.55)
            _fill_visible_number(page, "URR (%)", 72)
            _fill_visible_number(page, "Albumin (g/L)", 40)
            _fill_visible_number(page, "Dialysis vintage (months)", 50)
            page.locator("button:visible").last.click()

            compact_text = _wait_for_visible_text(page, "Compact model result")
            _assert_result_text(compact_text, "Compact model result")

            page.locator('[role="tab"]').nth(1).click(force=True)
            page.wait_for_timeout(1500)

            _fill_visible_number(page, "Age (years)", 57)
            _fill_visible_number(page, "Dialysis vintage (months)", 48)
            _fill_visible_number(page, "Dry weight (kg)", 61.0)
            _fill_visible_number(page, "Hemoglobin (g/L)", 112)
            _fill_visible_number(page, "CRP (mg/L)", 4.0)
            _fill_visible_number(page, "TSAT (%)", 25)
            _fill_visible_number(page, "Current-quarter ERI", 18.5)
            page.locator("button:visible").last.click()

            formal_text = _wait_for_visible_text(page, "Full model result")
            _assert_result_text(formal_text, "Full model result")

            full_body = page.locator("body").inner_text()
            assert "Predicted risk" in full_body
            assert "Risk category" in full_body
            assert "High-risk threshold" in full_body

            browser.close()
    finally:
        app.stop()

    assert not page_errors, f"Unexpected page errors: {page_errors}"
    assert "Traceback" not in app.stderr, app.stderr
    assert "Uncaught app execution" not in app.stderr, app.stderr
