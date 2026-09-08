"""JSONL and lazy optional benchmark ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from .schema import TaskRecord


@dataclass(frozen=True)
class BenchmarkSpec:
    kind: str
    role: str
    path: Path | None = None
    revision: str | None = None
    split: str = "test"
    field_map: Mapping[str, str] = field(default_factory=dict)


_DATASETS = {
    "taco": ("BAAI/TACO", None),
    "livecodebench_v6": ("livecodebench/code_generation_lite", "v6"),
    "bigcodebench_full": ("bigcode/bigcodebench", "default"),
    "bigcodebench_hard": ("bigcode/bigcodebench", "hard"),
    "humaneval_plus": ("evalplus/humanevalplus", None),
    "mbpp_plus": ("evalplus/mbppplus", None),
}


def _record(row: Mapping[str, Any], spec: BenchmarkSpec) -> TaskRecord:
    mapping = {"task_id": "task_id", "statement": "statement", "starter_code": "starter_code", "tests": "tests", "source": "source", "lineage_group": "lineage_group"}
    mapping.update(spec.field_map)
    missing = [field for field, source in mapping.items() if source not in row]
    if missing:
        raise ValueError(f"benchmark row missing required fields: {missing}")
    tests = row[mapping["tests"]]
    if isinstance(tests, str):
        tests = (tests,)
    return TaskRecord(str(row[mapping["task_id"]]), str(row[mapping["statement"]]), str(row[mapping["starter_code"]]), tuple(str(item) for item in tests), str(row[mapping["source"]]), str(row[mapping["lineage_group"]]), spec.role)


def load_tasks(spec: BenchmarkSpec) -> tuple[TaskRecord, ...]:
    if spec.kind == "jsonl":
        if spec.path is None:
            raise ValueError("JSONL benchmark requires a path")
        try:
            rows = [json.loads(line) for line in spec.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not load JSONL benchmark: {exc}") from exc
    else:
        if spec.kind not in _DATASETS:
            raise ValueError(f"unknown benchmark kind: {spec.kind}")
        if not spec.revision:
            raise ValueError("remote benchmark requires an immutable revision")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("remote benchmarks require the optional benchmarks extra") from exc
        dataset, config = _DATASETS[spec.kind]
        kwargs = {"path": dataset, "split": spec.split, "revision": spec.revision}
        if config is not None:
            kwargs["name"] = config
        rows = list(load_dataset(**kwargs))
    tasks = tuple(_record(row, spec) for row in rows)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("benchmark task IDs must be unique")
    return tasks
