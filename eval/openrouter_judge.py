#!/usr/bin/env python3
"""Judge generated AL-QASIDA outputs with an OpenRouter-hosted LLM."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, *args, **kwargs):
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data_processing" / "data"
DEFAULT_GLOBAL_METRICS = REPO_ROOT / "llm_outputs" / "llm_judge_directory_metrics.jsonl"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SCORE_FIELDS = (
    "dialect_authenticity",
    "coherence",
    "arabic_fluency",
    "msa_formality",
)
SCORE_MIN = 1
SCORE_MAX = 5

SOURCE_TO_INPUT = {
    "BTEC": ("mono", "btec/madar26"),
    "FLORES": ("mono", "wiki/flores"),
    "HABIBI": ("mono", "music/habibi"),
    "TWEET": ("mono", "tweets/nadi2023"),
    "Cohere": ("xling", "hehe"),
    "Okapi": ("xling", "okapi"),
    "ShareGPT": ("xling", "sharegpt"),
}

DIALECT_NAMES = {
    "dza": "Algerian Arabic",
    "egy": "Egyptian Arabic",
    "kwt": "Kuwaiti Arabic",
    "lbn": "Lebanese Arabic",
    "mar": "Moroccan Arabic",
    "pse": "Palestinian Arabic",
    "sau": "Saudi Arabic",
    "sdn": "Sudanese Arabic",
    "syr": "Syrian Arabic",
}

STEERING_CITY_TO_DIALECT = {
    "beirut": "lbn",
    "cairo": "egy",
    "rabat": "mar",
    "riyadh": "sau",
}


@dataclass(frozen=True)
class SampleFile:
    path: Path
    source: str
    dialect: str
    input_path: Path | None


@dataclass(frozen=True)
class PendingSample:
    sample_file: SampleFile
    row_index: int
    output: str


@dataclass(frozen=True)
class TargetDialect:
    code: str
    name: str
    source: str
    steering_city: str | None = None


class JudgeError(RuntimeError):
    """Raised when the judge response cannot be used."""


def parse_sample_filename(path: Path) -> tuple[str, str] | None:
    """Return (source, dialect) for supported *_samples.csv filenames."""
    match = re.fullmatch(
        r"DialectID_(?P<source>.+)_(?P<dialect>[a-z]{3})_samples",
        path.stem,
    )
    if not match:
        return None
    return match.group("source"), match.group("dialect")


def prompt_path_for(data_dir: Path, source: str, dialect: str) -> Path | None:
    source_info = SOURCE_TO_INPUT.get(source)
    if source_info is None:
        return None
    xtext, subdir = source_info
    return data_dir / xtext / subdir / f"{dialect}.csv"


def parse_directory_target(output_dir: Path) -> TargetDialect | None:
    match = re.search(r"(?:^|_)steered_(?P<city>[a-z]+)(?:_|$)", output_dir.name)
    if match is None:
        return None
    city = match.group("city")
    code = STEERING_CITY_TO_DIALECT.get(city)
    if code is None:
        return None
    return TargetDialect(
        code=code,
        name=DIALECT_NAMES.get(code, code),
        source="directory",
        steering_city=city,
    )


def sample_target(output_dir: Path, sample_file: SampleFile) -> TargetDialect:
    directory_target = parse_directory_target(output_dir)
    if directory_target is not None:
        return directory_target
    return TargetDialect(
        code=sample_file.dialect,
        name=DIALECT_NAMES.get(sample_file.dialect, sample_file.dialect),
        source="file",
    )


def iter_sample_files(output_dir: Path, data_dir: Path) -> list[SampleFile]:
    sample_files: list[SampleFile] = []
    for path in sorted(output_dir.glob("*_samples.csv")):
        parsed = parse_sample_filename(path)
        if parsed is None:
            continue
        source, dialect = parsed
        sample_files.append(
            SampleFile(
                path=path,
                source=source,
                dialect=dialect,
                input_path=prompt_path_for(data_dir, source, dialect),
            )
        )
    return sample_files


def read_csv_column(path: Path, column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain a {column!r} column")
        return [row.get(column, "") for row in reader]


def read_outputs(path: Path) -> list[str]:
    return [text.replace("\\n", "\n") for text in read_csv_column(path, "text")]


def read_prompts(path: Path | None) -> list[str] | None:
    if path is None or not path.is_file():
        return None
    return read_csv_column(path, "prompt")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise JudgeError(f"judge did not return JSON: {text[:200]!r}")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise JudgeError("judge JSON response is not an object")
    return parsed


def normalize_scores(parsed: dict[str, Any]) -> dict[str, int]:
    scores = parsed.get("scores", parsed)
    if not isinstance(scores, dict):
        raise JudgeError("judge response has no score object")

    normalized: dict[str, int] = {}
    for field in SCORE_FIELDS:
        value = scores.get(field)
        if isinstance(value, bool):
            raise JudgeError(f"{field} must be numeric, got boolean")
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise JudgeError(f"{field} must be an integer, got {value!r}") from None
        if number < SCORE_MIN or number > SCORE_MAX:
            raise JudgeError(f"{field}={number} outside {SCORE_MIN}-{SCORE_MAX}")
        normalized[field] = number
    return normalized


def build_messages(
    *,
    dialect: str,
    source: str,
    prompt: str | None,
    output: str,
) -> list[dict[str, str]]:
    dialect_name = DIALECT_NAMES.get(dialect, dialect)
    prompt_block = prompt if prompt is not None else "PROMPT NOT AVAILABLE"
    prompt_instruction = (
        "When the prompt is unavailable, score coherence using only whether the "
        "output is internally sensible and complete."
        if prompt is None
        else "Use the prompt to judge whether the output is coherent and responsive."
    )

    system = (
        "You are an expert Arabic dialect evaluator. Judge the generated output "
        "for the requested target dialect. Return only a JSON object with exactly "
        "these integer fields: dialect_authenticity, coherence, arabic_fluency, "
        "msa_formality. Do not include explanations or any extra keys."
    )
    user = f"""Target dialect code: {dialect}
Target dialect name: {dialect_name}
Dataset/source: {source}

Scoring scale: integer 1 to 5.
- dialect_authenticity: 1 = not the target dialect, mostly MSA/another dialect, 5 = strongly natural target dialect.
- coherence: 1 = nonsensical or unrelated, 5 = fully sensible and responsive.
- arabic_fluency: 1 = broken/unnatural Arabic, 5 = natural, fluent Arabic.
- msa_formality: 1 = very colloquial/dialectal, 5 = very formal/MSA-like.

{prompt_instruction}

Prompt:
{prompt_block}

Generated output:
{output}

Return JSON only, for example:
{{"dialect_authenticity": 4, "coherence": 5, "arabic_fluency": 4, "msa_formality": 2}}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class OpenRouterJudge:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float,
        retries: int,
        retry_sleep: float,
        app_title: str | None,
        referer: str | None,
        use_response_format: bool,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.app_title = app_title
        self.referer = referer
        self.use_response_format = use_response_format

    def headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        return headers

    def payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens": 128,
        }
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def judge(
        self,
        *,
        dialect: str,
        source: str,
        prompt: str | None,
        output: str,
    ) -> tuple[dict[str, int], dict[str, Any] | None]:
        messages = build_messages(
            dialect=dialect,
            source=source,
            prompt=prompt,
            output=output,
        )
        payload = self.payload(messages)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    OPENROUTER_URL,
                    headers=self.headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 500 or response.status_code == 429:
                    raise JudgeError(
                        f"OpenRouter retryable HTTP {response.status_code}: {response.text[:300]}"
                    )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                scores = normalize_scores(extract_json_object(content))
                return scores, data.get("usage")
            except (requests.RequestException, KeyError, IndexError, ValueError, JudgeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_sleep * (attempt + 1))

        raise JudgeError(f"failed after {self.retries + 1} attempt(s): {last_error}")


def load_completed_keys(path: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            file_name = record.get("file")
            row_index = record.get("row_index")
            scores = record.get("scores")
            if isinstance(file_name, str) and isinstance(row_index, int) and isinstance(scores, dict):
                completed.add((file_name, row_index))
    return completed


def collect_pending_samples(
    *,
    sample_files: list[SampleFile],
    completed: set[tuple[str, int]],
    max_samples_per_file: int | None,
    sample_limit: int | None,
) -> list[PendingSample]:
    pending: list[PendingSample] = []
    for sample_file in sample_files:
        outputs = read_outputs(sample_file.path)
        if max_samples_per_file is not None:
            outputs = outputs[:max_samples_per_file]

        for row_index, output in enumerate(outputs):
            if (sample_file.path.name, row_index) in completed:
                continue
            pending.append(
                PendingSample(
                    sample_file=sample_file,
                    row_index=row_index,
                    output=output,
                )
            )
            if sample_limit is not None and len(pending) >= sample_limit:
                return pending
    return pending


def load_scored_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            scores = record.get("scores")
            if isinstance(scores, dict) and all(field in scores for field in SCORE_FIELDS):
                records.append(record)
    return records


def mean_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    means: dict[str, float] = {}
    for field in SCORE_FIELDS:
        values = [float(record["scores"][field]) for record in records]
        means[f"{field}_mean"] = round(statistics.fmean(values), 4) if values else 0.0
    return means


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n_samples": len(records), **mean_scores(records)}


def summarize_by(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = str(record.get(key, "unknown"))
        grouped.setdefault(value, []).append(record)
    return {value: summarize_group(items) for value, items in sorted(grouped.items())}


def build_directory_metrics(
    *,
    output_dir: Path,
    sample_jsonl: Path,
    judge_model: str,
) -> dict[str, Any]:
    records = load_scored_records(sample_jsonl)
    missing_prompt_count = sum(1 for record in records if not record.get("prompt_available"))
    directory_target = parse_directory_target(output_dir)
    return {
        "directory": output_dir.name,
        "directory_path": str(output_dir.resolve()),
        "judge_model": judge_model,
        "created_at_utc": now_utc(),
        "target_dialect": directory_target.code if directory_target else None,
        "target_dialect_name": directory_target.name if directory_target else None,
        "steering_city": directory_target.steering_city if directory_target else None,
        "n_samples": len(records),
        "missing_prompt_count": missing_prompt_count,
        "overall": summarize_group(records),
        "by_source": summarize_by(records, "source"),
        "by_dialect": summarize_by(records, "dialect"),
        "by_target_dialect": summarize_by(records, "target_dialect"),
        "by_file_dialect": summarize_by(records, "file_dialect"),
        "by_file": summarize_by(records, "file"),
    }


def write_single_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        outfile.write(json_dumps(record) + "\n")


def upsert_global_metrics(path: Path, metrics: list[dict[str, Any]]) -> None:
    existing: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("directory_path")
                if isinstance(key, str):
                    existing[key] = record

    for record in metrics:
        existing[str(record["directory_path"])] = record

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for key in sorted(existing):
            outfile.write(json_dumps(existing[key]) + "\n")


def select_prompt(
    *,
    prompts: list[str] | None,
    row_index: int,
    missing_policy: str,
    prompt_path: Path | None,
) -> tuple[str | None, bool]:
    if prompts is None or row_index >= len(prompts):
        if missing_policy == "error":
            detail = str(prompt_path) if prompt_path is not None else "unmapped prompt path"
            raise FileNotFoundError(f"missing prompt row/file for {detail} row {row_index}")
        if missing_policy == "skip":
            return None, False
        return None, False
    return prompts[row_index], True


def judge_directory(
    *,
    output_dir: Path,
    data_dir: Path,
    judge: OpenRouterJudge,
    sample_jsonl_name: str,
    metrics_jsonl_name: str,
    force: bool,
    missing_prompt_policy: str,
    max_samples_per_file: int | None,
    sample_limit: int | None,
    sleep: float,
    show_progress: bool,
    dry_run: bool,
) -> dict[str, Any] | None:
    sample_files = iter_sample_files(output_dir, data_dir)
    if not sample_files:
        print(f"warning: no supported *_samples.csv files in {output_dir}", file=sys.stderr)
        return None

    sample_jsonl = output_dir / sample_jsonl_name
    metrics_jsonl = output_dir / metrics_jsonl_name
    completed = set() if force else load_completed_keys(sample_jsonl)
    if force and sample_jsonl.exists() and not dry_run:
        sample_jsonl.unlink()

    directory_target = parse_directory_target(output_dir)
    missing_prompt_files = [
        sample_file
        for sample_file in sample_files
        if sample_file.input_path is None or not sample_file.input_path.is_file()
    ]
    pending_samples = collect_pending_samples(
        sample_files=sample_files,
        completed=completed,
        max_samples_per_file=max_samples_per_file,
        sample_limit=sample_limit,
    )

    if dry_run:
        print(
            f"{output_dir}: would judge {len(pending_samples)} new sample(s) "
            f"across {len(sample_files)} file(s)"
        )
        if directory_target is not None:
            print(
                "dialect_authenticity target: "
                f"{directory_target.name} from steering city {directory_target.steering_city!r}"
            )
        if missing_prompt_files:
            print(
                f"warning: {len(missing_prompt_files)} file(s) have no matching prompt CSV",
                file=sys.stderr,
            )
            for sample_file in missing_prompt_files[:10]:
                print(
                    f"  {sample_file.path.name} -> {sample_file.input_path}",
                    file=sys.stderr,
                )
            if len(missing_prompt_files) > 10:
                print("  ...", file=sys.stderr)
        return None

    sample_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if directory_target is not None:
        print(
            "dialect_authenticity target: "
            f"{directory_target.name} from steering city {directory_target.steering_city!r}"
        )
    judged_count = 0
    skipped_count = 0
    prompt_cache: dict[str, list[str] | None] = {}
    warned_missing_prompt_files: set[str] = set()
    progress_iter = tqdm(
        pending_samples,
        desc=output_dir.name,
        unit="sample",
        disable=not show_progress,
    )
    with sample_jsonl.open("a", encoding="utf-8") as outfile:
        for pending in progress_iter:
            sample_file = pending.sample_file
            prompt_cache_key = sample_file.path.name
            if prompt_cache_key not in prompt_cache:
                prompt_cache[prompt_cache_key] = read_prompts(sample_file.input_path)
            prompts = prompt_cache[prompt_cache_key]
            if prompts is None and missing_prompt_policy != "skip":
                if sample_file.path.name not in warned_missing_prompt_files:
                    print(
                        f"warning: prompt CSV not found for {sample_file.path.name}: "
                        f"{sample_file.input_path}",
                        file=sys.stderr,
                    )
                    warned_missing_prompt_files.add(sample_file.path.name)

            prompt, prompt_available = select_prompt(
                prompts=prompts,
                row_index=pending.row_index,
                missing_policy=missing_prompt_policy,
                prompt_path=sample_file.input_path,
            )
            if not prompt_available and missing_prompt_policy == "skip":
                skipped_count += 1
                continue

            target = sample_target(output_dir, sample_file)
            scores, usage = judge.judge(
                dialect=target.code,
                source=sample_file.source,
                prompt=prompt,
                output=pending.output,
            )
            record = {
                "directory": output_dir.name,
                "directory_path": str(output_dir.resolve()),
                "file": sample_file.path.name,
                "source": sample_file.source,
                "dialect": target.code,
                "target_dialect": target.code,
                "target_dialect_name": target.name,
                "target_dialect_source": target.source,
                "steering_dialect": target.code if target.source == "directory" else None,
                "steering_dialect_name": target.name if target.source == "directory" else None,
                "steering_city": target.steering_city,
                "file_dialect": sample_file.dialect,
                "file_dialect_name": DIALECT_NAMES.get(sample_file.dialect, sample_file.dialect),
                "row_index": pending.row_index,
                "prompt": prompt,
                "prompt_available": prompt_available,
                "output": pending.output,
                "scores": scores,
                "judge_model": judge.model,
                "judged_at_utc": now_utc(),
            }
            if usage is not None:
                record["usage"] = usage
            outfile.write(json_dumps(record) + "\n")
            outfile.flush()
            judged_count += 1
            if sleep:
                time.sleep(sleep)

    metrics = build_directory_metrics(
        output_dir=output_dir,
        sample_jsonl=sample_jsonl,
        judge_model=judge.model,
    )
    write_single_jsonl(metrics_jsonl, metrics)
    print(
        f"{output_dir}: judged {judged_count} new sample(s), "
        f"skipped {skipped_count}, metrics -> {metrics_jsonl}"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge scoring over AL-QASIDA *_samples.csv outputs."
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="One or more llm_outputs result directories to judge.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root AL-QASIDA data directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4o-mini",
        help="OpenRouter model id to use as judge.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    parser.add_argument(
        "--sample-jsonl-name",
        default="llm_judge_samples.jsonl",
        help="Per-directory JSONL filename for per-sample scores.",
    )
    parser.add_argument(
        "--metrics-jsonl-name",
        default="llm_judge_metrics.jsonl",
        help="Per-directory JSONL filename for aggregate metrics.",
    )
    parser.add_argument(
        "--global-metrics-jsonl",
        type=Path,
        default=DEFAULT_GLOBAL_METRICS,
        help="Global JSONL path with one metrics record per judged directory.",
    )
    parser.add_argument(
        "--missing-prompt-policy",
        choices=("warn", "skip", "error"),
        default="warn",
        help="How to handle missing prompt CSVs or rows. Default judges without prompt and warns.",
    )
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=None,
        help="Limit judged rows from each *_samples.csv file, useful for smoke tests.",
    )
    parser.add_argument(
        "--sample-limit",
        "--limit-samples",
        dest="sample_limit",
        type=int,
        default=None,
        help="Limit total pending samples judged per directory, useful for quick tests.",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries for parse/HTTP failures.")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Base retry sleep in seconds.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between successful API calls.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-sample JSONL.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without API calls.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        help="Do not send OpenRouter response_format=json_object.",
    )
    parser.add_argument(
        "--app-title",
        default="AL-QASIDA LLM Judge",
        help="Optional X-OpenRouter-Title header.",
    )
    parser.add_argument(
        "--referer",
        default=None,
        help="Optional HTTP-Referer header for OpenRouter app attribution.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.max_samples_per_file is not None and args.max_samples_per_file < 1:
        print("error: --max-samples-per-file must be positive", file=sys.stderr)
        return 2
    if args.sample_limit is not None and args.sample_limit < 1:
        print("error: --sample-limit must be positive", file=sys.stderr)
        return 2

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        print(f"error: set {args.api_key_env} before running the judge", file=sys.stderr)
        return 2

    judge = OpenRouterJudge(
        api_key=api_key or "",
        model=args.judge_model,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        app_title=args.app_title,
        referer=args.referer,
        use_response_format=not args.no_response_format,
    )

    metrics_records: list[dict[str, Any]] = []
    for directory in args.directories:
        output_dir = directory.expanduser().resolve()
        if not output_dir.is_dir():
            print(f"error: {output_dir} is not a directory", file=sys.stderr)
            return 2
        metrics = judge_directory(
            output_dir=output_dir,
            data_dir=args.data_dir.expanduser().resolve(),
            judge=judge,
            sample_jsonl_name=args.sample_jsonl_name,
            metrics_jsonl_name=args.metrics_jsonl_name,
            force=args.force,
            missing_prompt_policy=args.missing_prompt_policy,
            max_samples_per_file=args.max_samples_per_file,
            sample_limit=args.sample_limit,
            sleep=args.sleep,
            show_progress=not args.no_progress,
            dry_run=args.dry_run,
        )
        if metrics is not None:
            metrics_records.append(metrics)

    if metrics_records and args.global_metrics_jsonl:
        upsert_global_metrics(args.global_metrics_jsonl.expanduser().resolve(), metrics_records)
        print(f"updated global metrics -> {args.global_metrics_jsonl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
