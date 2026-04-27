#!/usr/bin/env python3
"""Plot metric deltas for multiple llm_outputs directories against a reference."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from html import escape
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


KEY_COLUMNS = ("dataset", "dialect_code")
DEFAULT_SKIP_DIRS = {"pkls"}
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def summary_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path if path.is_file() else path / "summary.csv"


def read_summary(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise ValueError(f"{path} does not exist")

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
        base_key = (row["dataset"], row["dialect_code"])
        occurrence = counts[base_key]
        counts[base_key] += 1
        indexed[(*base_key, occurrence)] = row

    return indexed


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_score_column(column: str) -> bool:
    return column == "score" or column.endswith("_score")


def score_columns(field_sets: list[list[str]], requested: list[str] | None) -> list[str]:
    common = set(field_sets[0])
    for fields in field_sets[1:]:
        common &= set(fields)

    if requested:
        missing = [column for column in requested if column not in common]
        if missing:
            raise ValueError(f"requested metric column(s) missing from at least one summary: {', '.join(missing)}")
        return requested

    metrics = [column for column in field_sets[0] if column in common and is_score_column(column)]
    if not metrics:
        raise ValueError("no common score columns found; pass --metrics explicitly")
    return metrics


def mean_metric_delta(
    target_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    metric: str,
) -> tuple[float, int, int]:
    target_index = indexed_rows(target_rows)
    reference_index = indexed_rows(reference_rows)
    deltas = []
    skipped = 0

    for key, target_row in target_index.items():
        reference_row = reference_index.get(key)
        if reference_row is None:
            continue

        target_value = parse_float(target_row.get(metric))
        reference_value = parse_float(reference_row.get(metric))
        if target_value is None or reference_value is None:
            skipped += 1
            continue
        deltas.append(target_value - reference_value)

    if not deltas:
        raise ValueError(f"no paired numeric values found for metric '{metric}'")

    return sum(deltas) / len(deltas), len(deltas), skipped


def last_number(name: str) -> float | None:
    matches = NUMBER_RE.findall(name)
    return float(matches[-1]) if matches else None


def x_values(paths: list[Path], mode: str) -> tuple[list[float], list[str], bool]:
    names = [path.name for path in paths]
    if mode == "name":
        return list(range(len(names))), names, False

    numbers = [last_number(name) for name in names]
    if mode == "last-number" and any(number is None for number in numbers):
        missing = [name for name, number in zip(names, numbers) if number is None]
        raise ValueError(f"could not find a number in: {', '.join(missing)}")

    if mode == "auto" and any(number is None for number in numbers):
        return list(range(len(names))), names, False

    assert all(number is not None for number in numbers)
    return [float(number) for number in numbers if number is not None], [f"{number:g}" for number in numbers], True


def collect_target_dirs(args: argparse.Namespace, base_dir: Path) -> list[Path]:
    if args.all:
        reference = summary_path(args.reference).parent.resolve()
        dirs = [
            child.resolve()
            for child in base_dir.iterdir()
            if child.is_dir()
            and child.name not in DEFAULT_SKIP_DIRS
            and child.resolve() != reference
            and (child / "summary.csv").exists()
        ]
    else:
        dirs = [summary_path(path).parent for path in args.directories]

    if not dirs:
        raise ValueError("no target directories supplied")

    seen = set()
    unique_dirs = []
    for directory in dirs:
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(resolved)
    return unique_dirs


def sort_dirs(paths: list[Path], x_mode: str) -> list[Path]:
    if x_mode == "name":
        return sorted(paths, key=lambda path: path.name)

    keyed = [(last_number(path.name), path.name, path) for path in paths]
    if x_mode == "last-number" or all(number is not None for number, _, _ in keyed):
        return [path for number, _, path in sorted(keyed, key=lambda item: (item[0], item[1]))]
    return sorted(paths, key=lambda path: path.name)


def plot_with_matplotlib(
    output_path: Path,
    reference_dir: Path,
    target_dirs: list[Path],
    metrics: list[str],
    deltas: dict[str, list[float]],
    x_mode: str,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    xs, labels, numeric_axis = x_values(target_dirs, x_mode)
    fig_width = max(8, len(target_dirs) * 1.25)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    for metric in metrics:
        ax.plot(xs, deltas[metric], marker="o", linewidth=1.8, label=metric)

    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"Summary deltas vs {reference_dir.name}")
    ax.set_xlabel("Directory" if not numeric_axis else "Last number in directory name")
    ax.set_ylabel("Mean target-reference delta")
    ax.legend()

    if numeric_axis:
        ax.set_xticks(xs)
    else:
        ax.set_xticks(xs, labels, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_svg(
    output_path: Path,
    reference_dir: Path,
    target_dirs: list[Path],
    metrics: list[str],
    deltas: dict[str, list[float]],
    x_mode: str,
) -> None:
    xs, labels, numeric_axis = x_values(target_dirs, x_mode)
    min_x = min(xs)
    max_x = max(xs)
    if min_x == max_x:
        min_x -= 0.5
        max_x += 0.5

    all_values = [value for metric in metrics for value in deltas[metric]]
    max_abs_y = max(abs(value) for value in all_values) or 1.0
    y_limit = max_abs_y * 1.15

    width = max(900, 120 + len(target_dirs) * 90)
    height = 560
    left = 90
    right = 190
    top = 65
    bottom = 115
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]

    def x_pos(value: float) -> float:
        return left + ((value - min_x) / (max_x - min_x)) * plot_width

    def y_pos(value: float) -> float:
        return top + ((y_limit - value) / (2 * y_limit)) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: DejaVu Sans, Arial, sans-serif; fill: #222; }",
        ".title { font-size: 18px; font-weight: 700; }",
        ".label { font-size: 12px; }",
        ".tick { font-size: 10px; }",
        ".axis { stroke: #222; stroke-width: 1; }",
        ".grid { stroke: #d9d9d9; stroke-width: 1; }",
        "</style>",
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">Summary deltas vs {escape(reference_dir.name)}</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" />',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" />',
    ]

    for grid_value in (-y_limit, -y_limit / 2, 0, y_limit / 2, y_limit):
        y = y_pos(grid_value)
        class_name = "axis" if math.isclose(grid_value, 0.0) else "grid"
        svg.append(f'<line class="{class_name}" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" />')
        svg.append(f'<text class="tick" x="{left - 8}" y="{y + 3:.2f}" text-anchor="end">{grid_value:.4g}</text>')

    for x, label in zip(xs, labels):
        px = x_pos(x)
        svg.append(f'<line class="grid" x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_height}" />')
        svg.append(
            f'<text class="tick" transform="translate({px:.2f} {top + plot_height + 18}) rotate(-35)" '
            f'text-anchor="end">{escape(label)}</text>'
        )

    for metric_index, metric in enumerate(metrics):
        color = colors[metric_index % len(colors)]
        points = " ".join(
            f"{x_pos(x):.2f},{y_pos(value):.2f}"
            for x, value in zip(xs, deltas[metric])
        )
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" />')
        for x, value in zip(xs, deltas[metric]):
            svg.append(f'<circle cx="{x_pos(x):.2f}" cy="{y_pos(value):.2f}" r="3.5" fill="{color}" />')

        legend_y = top + metric_index * 22
        svg.append(f'<line x1="{width - right + 40}" y1="{legend_y}" x2="{width - right + 62}" y2="{legend_y}" stroke="{color}" stroke-width="2" />')
        svg.append(f'<text class="label" x="{width - right + 70}" y="{legend_y + 4}">{escape(metric)}</text>')

    x_label = "Last number in directory name" if numeric_axis else "Directory"
    svg.append(f'<text class="label" x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle">{x_label}</text>')
    svg.append(
        f'<text class="label" transform="translate(20 {top + plot_height / 2}) rotate(-90)" '
        f'text-anchor="middle">Mean target-reference delta</text>'
    )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def default_output_path(reference_dir: Path) -> Path:
    suffix = ".png" if plt is not None else ".svg"
    return reference_dir.parent / f"summary_deltas_vs_{reference_dir.name}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare several llm_outputs directories to a reference directory and plot "
            "one line per score column."
        )
    )
    parser.add_argument("directories", nargs="*", type=Path, help="Target directories, or paths to their summary.csv files.")
    parser.add_argument("--reference", "-r", required=True, type=Path, help="Reference directory, or its summary.csv.")
    parser.add_argument("--output", "-o", type=Path, help="Output plot path. Defaults beside the reference directory.")
    parser.add_argument("--metrics", "-m", nargs="+", help="Metric columns to plot. Defaults to common score columns.")
    parser.add_argument(
        "--x",
        choices=("auto", "name", "last-number"),
        default="auto",
        help="Use directory names or the final number in each directory name on the x-axis. Default: auto.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use every sibling directory with a summary.csv except pkls and the reference directory.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    reference_summary = summary_path(args.reference)
    reference_dir = reference_summary.parent
    output_path = args.output.expanduser().resolve() if args.output else default_output_path(reference_dir)

    try:
        target_dirs = sort_dirs(collect_target_dirs(args, base_dir), args.x)
        reference_rows, reference_fields = read_summary(reference_summary)

        target_data = []
        field_sets = [reference_fields]
        for directory in target_dirs:
            rows, fields = read_summary(directory / "summary.csv")
            target_data.append((directory, rows))
            field_sets.append(fields)

        metrics = score_columns(field_sets, args.metrics)
        deltas = {metric: [] for metric in metrics}
        skipped_total = Counter()
        paired_counts = Counter()

        for directory, rows in target_data:
            for metric in metrics:
                delta, paired_count, skipped = mean_metric_delta(rows, reference_rows, metric)
                deltas[metric].append(delta)
                paired_counts[metric] += paired_count
                skipped_total[metric] += skipped

        if output_path.suffix.lower() == ".svg" or plt is None:
            if output_path.suffix.lower() != ".svg":
                output_path = output_path.with_suffix(".svg")
            plot_svg(output_path, reference_dir, target_dirs, metrics, deltas, args.x)
        else:
            plot_with_matplotlib(output_path, reference_dir, target_dirs, metrics, deltas, args.x)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote delta plot to {output_path}")
    print(f"Compared {len(target_dirs)} target director{'y' if len(target_dirs) == 1 else 'ies'} against {reference_dir}")
    print(f"Plotted metrics: {', '.join(metrics)}")
    for metric in metrics:
        if skipped_total[metric]:
            print(f"Skipped {skipped_total[metric]} row(s) with missing/non-numeric {metric} values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
