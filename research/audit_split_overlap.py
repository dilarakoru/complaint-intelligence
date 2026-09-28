"""Audit identity and normalized-text overlap between historical split CSVs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import normalize_text


def overlap_report(train_csv: Path, test_csv: Path) -> dict:
    train = pd.read_csv(train_csv, dtype={"complaint_id": str})
    test = pd.read_csv(test_csv, dtype={"complaint_id": str})

    required = {"complaint_id", "complaint_what_happened"}
    for name, frame in (("train", train), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    train_ids = set(train["complaint_id"].dropna().astype(str))
    test_ids = set(test["complaint_id"].dropna().astype(str))

    train_text = set(train["complaint_what_happened"].fillna("").map(normalize_text))
    test_text = test["complaint_what_happened"].fillna("").map(normalize_text)

    id_overlap_rows = int(test["complaint_id"].astype(str).isin(train_ids).sum())
    text_overlap_rows = int(test_text.isin(train_text).sum())

    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_rows_with_train_id": id_overlap_rows,
        "test_rows_with_train_normalized_text": text_overlap_rows,
        "unique_id_overlap": int(len(train_ids & test_ids)),
        "unique_normalized_text_overlap": int(len(train_text & set(test_text))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure duplicate leakage across historical train/test CSVs.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(overlap_report(args.train, args.test), indent=2))


if __name__ == "__main__":
    main()
