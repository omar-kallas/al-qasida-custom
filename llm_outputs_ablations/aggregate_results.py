#!/usr/bin/env python3
"""Aggregate result metric CSVs into a summary.csv file."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


DIALECT_CODES = {"dza", "egy", "kwt", "mar", "pse", "sau", "sdn", "syr"}
NON_DIALECT_LANGS = {"eng", "msa"}


def parse_filename(path: Path) -> tuple[str, str]:
    """Return (dataset, dialect_code) inferred from a metrics filename."""
    stem = path.stem
    id_match = re.fullmatch(r"DialectID_(?P<dataset>.+)_(?P<dialect>[a-z]{3})_metrics", stem)
    if id_match:
        return id_match.group("dataset"), id_match.group("dialect")

    mt_match = re.fullmatch(
        r"DialectMT_(?P<dataset>.+)_(?P<src>[a-z]{3})-(?P<tgt>[a-z]{3})_metrics",
        stem,
    )
    if mt_match:
        dataset = mt_match.group("dataset")
        src = mt_match.group("src")
        tgt = mt_match.group("tgt")

        if (src in DIALECT_CODES and tgt in NON_DIALECT_LANGS) or (tgt in DIALECT_CODES and src in NON_DIALECT_LANGS):
            return dataset, f"{src}-{tgt}"

    raise ValueError(f"Cannot infer dataset and dialect code from {path.name}")


def iter_metric_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*_metrics.csv")
        if path.name != "summary_metrics.csv" and path.is_file()
    )


def aggregate(input_dir: Path) -> tuple[Path, int, list[Path]]:
    rows: list[dict[str, str]] = []
    fieldnames = ["dataset", "dialect_code"]
    skipped: list[Path] = []

    for path in iter_metric_files(input_dir):
        try:
            dataset, dialect_code = parse_filename(path)
        except ValueError:
            skipped.append(path)
            continue

        with path.open(newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if reader.fieldnames is None:
                continue

            for name in reader.fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)

            for row in reader:
                rows.append({"dataset": dataset, "dialect_code": dialect_code, **row})

    if not rows:
        raise ValueError(f"No metric rows found under {input_dir}")

    output_path = input_dir / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return output_path, len(rows), skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate *_metrics.csv result files into summary.csv."
    )
    parser.add_argument("directory", type=Path, help="Directory containing result CSV files.")
    args = parser.parse_args()

    input_dir = args.directory.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"error: {input_dir} is not a directory", file=sys.stderr)
        return 2

    try:
        output_path, row_count, skipped = aggregate(input_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {row_count} rows to {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with unrecognized names:", file=sys.stderr)
        for path in skipped:
            print(f"  {path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
