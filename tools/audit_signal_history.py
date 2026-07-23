#!/usr/bin/env python3
"""Detect changes to previously published NeuralTrend market/signal rows.

The manifest hashes the validated Date, Close, and epoch_signal columns. During
``--check``, each current CSV is truncated to the previously published last date
before hashing. Appending new rows is therefore allowed, while edits, deletions,
or reordered/deduplicated changes to older rows are flagged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "signal_history_manifest.json"


def load_validated_rows(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        csv_path,
        usecols=["Date", "Close", "epoch_signal"],
        parse_dates=["Date"],
    )
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame["epoch_signal"] = pd.to_numeric(frame["epoch_signal"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close", "epoch_signal"]).copy()

    finite_close = frame["Close"].map(lambda value: math.isfinite(float(value)))
    finite_signal = frame["epoch_signal"].map(lambda value: math.isfinite(float(value)))
    valid_signal = frame["epoch_signal"].isin([-1, 0, 1])

    frame = frame[
        finite_close
        & finite_signal
        & valid_signal
        & (frame["Close"] > 0)
    ].copy()
    frame["epoch_signal"] = frame["epoch_signal"].astype(int)
    frame = frame.sort_values("Date")
    frame = frame.drop_duplicates(subset=["Date"], keep="last")
    return frame[["Date", "Close", "epoch_signal"]]


def hash_rows(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()

    for row in frame.itertuples(index=False):
        date_text = pd.Timestamp(row.Date).date().isoformat()
        close_text = format(float(row.Close), ".17g")
        signal_text = str(int(row.epoch_signal))
        digest.update(f"{date_text},{close_text},{signal_text}\n".encode("utf-8"))

    return digest.hexdigest()


def build_entry(csv_path: Path, *, through_date: str | None = None) -> dict:
    frame = load_validated_rows(csv_path)

    if through_date:
        cutoff = pd.Timestamp(through_date)
        frame = frame[frame["Date"] <= cutoff].copy()

    if frame.empty:
        return {
            "row_count": 0,
            "first_date": None,
            "last_date": None,
            "content_sha256": hash_rows(frame),
        }

    return {
        "row_count": int(len(frame)),
        "first_date": frame["Date"].iloc[0].date().isoformat(),
        "last_date": frame["Date"].iloc[-1].date().isoformat(),
        "content_sha256": hash_rows(frame),
    }


def build_manifest(data_dir: Path) -> dict:
    files = {}
    skipped_files = []

    for csv_path in sorted(data_dir.glob("epoch_*.csv")):
        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
        except Exception as error:
            skipped_files.append({"filename": csv_path.name, "reason": str(error)[:300]})
            continue

        required_columns = {"Date", "Close", "epoch_signal"}
        if not required_columns.issubset(columns):
            skipped_files.append({
                "filename": csv_path.name,
                "reason": "Not an asset signal-history CSV",
            })
            continue

        files[csv_path.name] = build_entry(csv_path)

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_columns": ["Date", "Close", "epoch_signal"],
        "files": files,
        "skipped_files": skipped_files,
    }


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
        raise ValueError("Manifest has an invalid structure.")

    return payload


def check_history(data_dir: Path, manifest_path: Path) -> int:
    try:
        previous = load_manifest(manifest_path)
    except FileNotFoundError:
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: could not read manifest: {error}", file=sys.stderr)
        return 2

    problems = []
    previous_files = previous["files"]

    for filename, prior_entry in sorted(previous_files.items()):
        csv_path = data_dir / filename

        if not csv_path.is_file():
            problems.append(f"REMOVED FILE: {filename}")
            continue

        prior_last_date = prior_entry.get("last_date")

        try:
            current_prior_window = build_entry(
                csv_path,
                through_date=prior_last_date,
            )
        except Exception as error:  # deliberate operator-facing audit output
            problems.append(f"UNREADABLE: {filename}: {error}")
            continue

        if current_prior_window["content_sha256"] != prior_entry.get("content_sha256"):
            problems.append(
                "HISTORICAL CHANGE: "
                f"{filename} through {prior_last_date or 'empty history'}"
            )

    current_filenames = set()
    required_columns = {"Date", "Close", "epoch_signal"}
    for path in data_dir.glob("epoch_*.csv"):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if required_columns.issubset(columns):
            current_filenames.add(path.name)

    new_files = sorted(current_filenames - set(previous_files))

    if problems:
        print("Signal-history audit FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nReview each change and add a public_model_change_log.json entry "
            "before accepting intentional historical revisions."
        )
        return 1

    print(
        "Signal-history audit passed: previously published Date/Close/signal "
        "rows are unchanged."
    )
    if new_files:
        print(f"New asset files detected (allowed): {', '.join(new_files)}")
    return 0


def write_manifest(data_dir: Path, manifest_path: Path) -> int:
    manifest = build_manifest(data_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(manifest['files'])} file entries to {manifest_path}."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    manifest_path = args.manifest.resolve()

    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        return 2

    if args.check:
        return check_history(data_dir, manifest_path)

    return write_manifest(data_dir, manifest_path)


if __name__ == "__main__":
    raise SystemExit(main())
