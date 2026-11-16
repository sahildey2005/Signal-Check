"""Run Signal Check locally with a small SQLite feedback API.

This development server binds to 127.0.0.1 only. Do not expose it directly
on the public internet; use a production WSGI/ASGI host for deployment.
"""

from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import urlparse
import json
import os
from contextlib import contextmanager
import re
import sqlite3
import webbrowser


PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "feedback.sqlite3"
MAX_BODY_BYTES = 16_384
ALLOWED_ORIGIN = os.environ.get("SIGNAL_CHECK_ALLOWED_ORIGIN", "").strip()


@contextmanager
def connect_db():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with connect_db() as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                headline TEXT NOT NULL,
                clickbait_label INTEGER CHECK (clickbait_label IN (0,1) OR clickbait_label IS NULL),
                fake_real_label INTEGER CHECK (fake_real_label IN (0,1) OR fake_real_label IS NULL),
                evidence_url TEXT NOT NULL DEFAULT '',
                predicted_clickbait_score REAL,
                predicted_fake_real_score REAL,
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
                review_note TEXT NOT NULL DEFAULT ''
            )"""
        )
        db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)")


def valid_label(value: object) -> int | None:
    if value in (None, ""):
        return None
    if value in (0, "0", False):
        return 0
    if value in (1, "1", True):
        return 1
    raise ValueError("Labels must be 0, 1, or empty.")


class SignalCheckHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_DIR), **kwargs)

    def end_headers(self) -> None:
        origin = self.headers.get("Origin", "")
        if ALLOWED_ORIGIN and origin == ALLOWED_ORIGIN:
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/feedback.sqlite3"):
            self.send_json(404, {"ok": False, "error": "Not found."})
            return
        if path == "/api/status":
            with connect_db() as db:
                rows = db.execute(
                    "SELECT status, COUNT(*) AS total, "
                    "SUM(clickbait_label IS NOT NULL) AS clickbait_total, "
                    "SUM(fake_real_label IS NOT NULL) AS fake_real_total "
                    "FROM feedback GROUP BY status"
                ).fetchall()
            summary = {"pending": 0, "approved": 0, "rejected": 0,
                       "approved_clickbait_labels": 0, "approved_fake_real_labels": 0}
            for row in rows:
                summary[row["status"]] = row["total"] or 0
                if row["status"] == "approved":
                    summary["approved_clickbait_labels"] = row["clickbait_total"] or 0
                    summary["approved_fake_real_labels"] = row["fake_real_total"] or 0
            self.send_json(200, {"ok": True, **summary})
            return
        if path.startswith("/api/"):
            self.send_json(404, {"ok": False, "error": "API endpoint not found."})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/feedback":
            self.send_json(404, {"ok": False, "error": "API endpoint not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("Feedback request is empty or too large.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Feedback body must be a JSON object.")
            headline = re.sub(r"\s+", " ", str(payload.get("headline", ""))).strip()
            if not 12 <= len(headline) <= 280:
                raise ValueError("Headline must contain 12 to 280 characters.")
            clickbait_label = valid_label(payload.get("clickbait_label"))
            fake_real_label = valid_label(payload.get("fake_real_label"))
            if clickbait_label is None and fake_real_label is None:
                raise ValueError("Choose at least one model label before submitting.")
            evidence_url = str(payload.get("evidence_url", "")).strip()
            if fake_real_label is not None and not re.match(r"^https?://\S+$", evidence_url, re.I):
                raise ValueError("Add an http or https evidence link for a fake/true label.")
            predicted_clickbait = payload.get("predicted_clickbait_score")
            predicted_fake_real = payload.get("predicted_fake_real_score")
            for value in (predicted_clickbait, predicted_fake_real):
                if value is not None and not 0 <= float(value) <= 1:
                    raise ValueError("Prediction scores must be between 0 and 1.")
            with connect_db() as db:
                cursor = db.execute(
                    """INSERT INTO feedback
                    (headline, clickbait_label, fake_real_label, evidence_url,
                     predicted_clickbait_score, predicted_fake_real_score)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (headline, clickbait_label, fake_real_label, evidence_url,
                     predicted_clickbait, predicted_fake_real),
                )
                feedback_id = cursor.lastrowid
            self.send_json(201, {
                "ok": True,
                "id": feedback_id,
                "status": "pending",
                "message": "Thank you. Your labels are saved for review and are not used for training yet.",
            })
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self.send_json(400, {"ok": False, "error": str(error)})
        except Exception as error:  # keep internal details out of the browser
            print(f"Feedback API error: {error}")
            self.send_json(500, {"ok": False, "error": "The feedback could not be saved. Try again."})


def main() -> None:
    os.chdir(PROJECT_DIR)
    initialize_database()
    server = ThreadingHTTPServer(("127.0.0.1", 0), SignalCheckHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/index.html"
    print(f"Signal Check is running at {url}")
    print("Feedback is stored in feedback.sqlite3 and remains pending until reviewed.")
    print("Keep this window open while using the app. Press Ctrl+C here to stop it.")
    Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSignal Check stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
