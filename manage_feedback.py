"""Review user-submitted labels before they can enter model training.

Examples:
  python manage_feedback.py list
  python manage_feedback.py approve 12 --confirm-reviewed
  python manage_feedback.py reject 13 --note "No reliable supporting source"
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "feedback.sqlite3"
MIN_APPROVED_PER_MODEL = 10


def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SystemExit("No feedback database yet. Start run.bat and submit feedback first.")
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Signal Check feedback labels.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List pending feedback")
    approve = sub.add_parser("approve", help="Approve labels after checking them")
    approve.add_argument("feedback_id", type=int)
    approve.add_argument("--confirm-reviewed", action="store_true", required=True,
                         help="Confirm you checked the labels and evidence")
    reject = sub.add_parser("reject", help="Reject a feedback submission")
    reject.add_argument("feedback_id", type=int)
    reject.add_argument("--note", default="Not approved after review")
    args = parser.parse_args()

    with connect() as db:
        if args.command == "list":
            rows = db.execute(
                "SELECT * FROM feedback WHERE status='pending' ORDER BY id"
            ).fetchall()
            if not rows:
                print("No pending feedback.")
            for row in rows:
                print(f"\nID: {row['id']} | submitted: {row['created_at']}")
                print(f"Headline: {row['headline']}")
                print(f"Clickbait label: {row['clickbait_label']} (1=clickbait, 0=not clickbait)")
                print(f"Fake/true dataset label: {row['fake_real_label']} (1=fake-labelled, 0=true-labelled)")
                print(f"Evidence: {row['evidence_url'] or '(not supplied)'}")
                print(f"Scores at submission: clickbait={row['predicted_clickbait_score']}, fake/true={row['predicted_fake_real_score']}")
        elif args.command == "approve":
            cursor = db.execute(
                "UPDATE feedback SET status='approved', review_note='Reviewed and approved' WHERE id=? AND status='pending'",
                (args.feedback_id,),
            )
            if cursor.rowcount != 1:
                raise SystemExit("Pending feedback ID not found.")
            print(f"Feedback {args.feedback_id} approved.")
            counts = db.execute(
                "SELECT SUM(clickbait_label IS NOT NULL), SUM(fake_real_label IS NOT NULL) "
                "FROM feedback WHERE status='approved'"
            ).fetchone()
            clickbait_count = int(counts[0] or 0)
            fake_real_count = int(counts[1] or 0)
            print(f"Approved labels: clickbait {clickbait_count}/{MIN_APPROVED_PER_MODEL}; fake/true {fake_real_count}/{MIN_APPROVED_PER_MODEL}.")
            datasets = [PROJECT_DIR / "data" / name for name in ("clickbait_data.csv", "Fake.csv", "True.csv")]
            if clickbait_count >= MIN_APPROVED_PER_MODEL and fake_real_count >= MIN_APPROVED_PER_MODEL and all(path.is_file() for path in datasets):
                candidate = PROJECT_DIR / "models_candidate.json"
                print("Enough reviewed labels are available. Building a new candidate model…")
                subprocess.run([
                    sys.executable, str(PROJECT_DIR / "train_models.py"),
                    "--feedback-db", str(DB_PATH), "--output", str(candidate),
                ], check=True)
                print("Candidate created. Review its metrics and test it before replacing models.json.")
            elif not all(path.is_file() for path in datasets):
                print("Automatic candidate training is waiting for the three source CSVs in the data folder.")
            else:
                print("Candidate training will start automatically after at least 10 approved labels exist for each model.")
        elif args.command == "reject":
            cursor = db.execute(
                "UPDATE feedback SET status='rejected', review_note=? WHERE id=? AND status='pending'",
                (args.note[:500], args.feedback_id),
            )
            if cursor.rowcount != 1:
                raise SystemExit("Pending feedback ID not found.")
            print(f"Feedback {args.feedback_id} rejected.")


if __name__ == "__main__":
    main()
