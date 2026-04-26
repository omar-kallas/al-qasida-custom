#!/usr/bin/env python3
"""Plot per-dialect metric deltas between two summary CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


KEY_COLUMNS = ("dataset", "dialect_code")


def read_summary(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")

        missing = [column for column in KEY_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

        return list(reader), reader.fieldnames


def indexed_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    indexed = {}

    for row in rows:
        dataset = row["dataset"]
        dialect_code = row["dialect_code"]
        base_key = (dataset, dialect_code)
        occurrence = counts[base_key]
        counts[base_key] += 1
        indexed[(dataset, dialect_code, occurrence)] = row

    return indexed


def validate_pairing(
    target_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str, int], dict[str, str]], dict[tuple[str, str, int], dict[str, str]]]:
    if len(target_rows) != len(reference_rows):
        raise ValueError(
            "summary files must have the same number of rows: "
            f"target has {len(target_rows)}, reference has {len(reference_rows)}"
        )

    target_counts = Counter((row["dataset"], row["dialect_code"]) for row in target_rows)
    reference_counts = Counter((row["dataset"], row["dialect_code"]) for row in reference_rows)
    if target_counts != reference_counts:
        target_only = target_counts - reference_counts
        reference_only = reference_counts - target_counts
        details = []
        if target_only:
            details.append(f"target-only combinations: {dict(target_only)}")
        if reference_only:
            details.append(f"reference-only combinations: {dict(reference_only)}")
        raise ValueError("summary files do not contain the same dataset/dialect combinations; " + "; ".join(details))

    return indexed_rows(target_rows), indexed_rows(reference_rows)


def parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compute_grouped_deltas(
    target_rows: dict[tuple[str, str, int], dict[str, str]],
    reference_rows: dict[tuple[str, str, int], dict[str, str]],
    metric: str,
) -> tuple[dict[str, dict[str, float]], int]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for key in sorted(target_rows):
        dataset, dialect_code, _ = key
        target_value = parse_float(target_rows[key].get(metric, ""))
        reference_value = parse_float(reference_rows[key].get(metric, ""))
        if target_value is None or reference_value is None:
            skipped += 1
            continue
        grouped[dialect_code][dataset].append(target_value - reference_value)

    averaged = {
        dialect: {
            dataset: sum(values) / len(values)
            for dataset, values in sorted(dataset_values.items())
        }
        for dialect, dataset_values in sorted(grouped.items())
    }
    return averaged, skipped


def plot_deltas(
    deltas: dict[str, dict[str, float]],
    metric: str,
    target_path: Path,
    reference_path: Path,
    output_path: Path,
) -> None:
    if plt is None or output_path.suffix.lower() == ".svg":
        plot_deltas_svg(deltas, metric, target_path, reference_path, output_path)
        return

    dialects = sorted(deltas)
    if not dialects:
        raise ValueError(f"no paired numeric values found for metric '{metric}'")

    cols = min(4, len(dialects))
    rows = math.ceil(len(dialects) / cols)
    fig_width = max(10, cols * 3.5)
    fig_height = max(4, rows * 3.2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    max_abs_delta = max(
        abs(delta)
        for dataset_values in deltas.values()
        for delta in dataset_values.values()
    )
    y_limit = max_abs_delta * 1.15 if max_abs_delta else 1.0

    for index, dialect in enumerate(dialects):
        ax = axes[index // cols][index % cols]
        dataset_values = deltas[dialect]
        datasets = list(dataset_values)
        values = [dataset_values[dataset] for dataset in datasets]
        colors = ["#1f77b4" if value >= 0 else "#d62728" for value in values]

        ax.bar(datasets, values, color=colors)
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_title(dialect)
        ax.set_ylim(-y_limit, y_limit)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=0.25)

        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    for index in range(len(dialects), rows * cols):
        axes[index // cols][index % cols].axis("off")

    fig.suptitle(target_path.parent.name, fontsize=13)
    fig.supxlabel("Dataset")
    fig.supylabel("Absolute change from reference")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_deltas_svg(
    deltas: dict[str, dict[str, float]],
    metric: str,
    target_path: Path,
    reference_path: Path,
    output_path: Path,
) -> None:
    dialects = sorted(deltas)
    if not dialects:
        raise ValueError(f"no paired numeric values found for metric '{metric}'")

    cols = min(4, len(dialects))
    rows = math.ceil(len(dialects) / cols)
    panel_width = 360
    panel_height = 280
    margin_top = 70
    margin_left = 70
    margin_right = 30
    margin_bottom = 95
    width = cols * panel_width
    height = rows * panel_height + margin_top

    max_abs_delta = max(
        abs(delta)
        for dataset_values in deltas.values()
        for delta in dataset_values.values()
    )
    y_limit = max_abs_delta * 1.15 if max_abs_delta else 1.0

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: DejaVu Sans, Arial, sans-serif; fill: #222; }",
        ".title { font-size: 18px; font-weight: 700; }",
        ".panel-title { font-size: 14px; font-weight: 700; }",
        ".tick { font-size: 10px; }",
        ".label { font-size: 12px; }",
        ".grid { stroke: #d9d9d9; stroke-width: 1; }",
        ".axis { stroke: #222; stroke-width: 1; }",
        "</style>",
        f'<text class="title" x="{width / 2}" y="28" text-anchor="middle">'
        f'{escape(target_path.parent.name)}</text>',
        f'<text class="label" x="{width / 2}" y="{height - 8}" text-anchor="middle">Dataset</text>',
        f'<text class="label" transform="translate(16 {height / 2}) rotate(-90)" '
        f'text-anchor="middle">Absolute change from reference</text>',
    ]

    for index, dialect in enumerate(dialects):
        col = index % cols
        row = index // cols
        x0 = col * panel_width + margin_left
        y0 = row * panel_height + margin_top + 25
        plot_width = panel_width - margin_left - margin_right
        plot_height = panel_height - margin_top - margin_bottom
        zero_y = y0 + plot_height / 2
        scale = (plot_height / 2) / y_limit

        dataset_values = deltas[dialect]
        datasets = list(dataset_values)
        values = [dataset_values[dataset] for dataset in datasets]
        slot_width = plot_width / max(1, len(datasets))
        bar_width = min(42, slot_width * 0.7)

        svg.append(
            f'<text class="panel-title" x="{x0 + plot_width / 2}" y="{y0 - 14}" '
            f'text-anchor="middle">{escape(dialect)}</text>'
        )

        for grid_value in (-y_limit, -y_limit / 2, 0, y_limit / 2, y_limit):
            y = zero_y - grid_value * scale
            class_name = "axis" if grid_value == 0 else "grid"
            svg.append(f'<line class="{class_name}" x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_width}" y2="{y:.2f}" />')
            svg.append(
                f'<text class="tick" x="{x0 - 6}" y="{y + 3:.2f}" text-anchor="end">'
                f'{grid_value:.3g}</text>'
            )

        svg.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_height}" />')

        for dataset, value_index in zip(datasets, range(len(datasets))):
            value = values[value_index]
            cx = x0 + slot_width * value_index + slot_width / 2
            bar_top = zero_y - max(value, 0) * scale
            bar_height = abs(value) * scale
            if value < 0:
                bar_top = zero_y
            color = "#1f77b4" if value >= 0 else "#d62728"
            svg.append(
                f'<rect x="{cx - bar_width / 2:.2f}" y="{bar_top:.2f}" '
                f'width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" />'
            )
            svg.append(
                f'<text class="tick" transform="translate({cx:.2f} {y0 + plot_height + 12:.2f}) rotate(-45)" '
                f'text-anchor="end">{escape(dataset)}</text>'
            )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def default_output_path(target_path: Path, metric: str) -> Path:
    safe_metric = metric.replace("/", "_").replace("%", "pct")
    suffix = ".png" if plt is not None else ".svg"
    return target_path.parent / f"{target_path.stem}_delta_{safe_metric}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot per-dialect metric deltas between a target and reference summary CSV."
    )
    parser.add_argument("target_summary", type=Path, help="Summary CSV to compare against the reference.")
    parser.add_argument("reference_summary", type=Path, help="Reference summary CSV.")
    parser.add_argument("--metric", default="score", help="Numeric metric column to compare. Default: score.")
    parser.add_argument("--output", type=Path, help="Output image path. Defaults next to target summary.")
    args = parser.parse_args()

    target_path = args.target_summary.expanduser().resolve()
    reference_path = args.reference_summary.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(target_path, args.metric)
    )
    if plt is None and output_path.suffix.lower() != ".svg":
        print(
            "error: matplotlib is not installed, so only SVG output is available; "
            "use an .svg output path",
            file=sys.stderr,
        )
        return 1

    try:
        target_rows, target_fields = read_summary(target_path)
        reference_rows, reference_fields = read_summary(reference_path)
        if args.metric not in target_fields:
            raise ValueError(f"{target_path} is missing metric column '{args.metric}'")
        if args.metric not in reference_fields:
            raise ValueError(f"{reference_path} is missing metric column '{args.metric}'")

        target_indexed, reference_indexed = validate_pairing(target_rows, reference_rows)
        deltas, skipped = compute_grouped_deltas(target_indexed, reference_indexed, args.metric)
        plot_deltas(deltas, args.metric, target_path, reference_path, output_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote delta plot to {output_path}")
    if skipped:
        print(f"Skipped {skipped} row(s) with missing or non-numeric '{args.metric}' values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
