"""Recursive, immutable experiment configuration and fail-closed validation."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import yaml

from .registry import ALLOCATOR_ARMS, DATASETS, DATASET_ROLES, HARDWARE_PROFILES, MODELS, PROVENANCE_MODES


class ConfigError(ValueError):
    """Configuration violates a registered experiment contract."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return copy.deepcopy(value)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError("configuration must be JSON-canonicalizable") from exc


def config_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class ExperimentConfig:
    data: Mapping[str, Any]
    sha256: str
    source_paths: tuple[Path, ...]
    source_hashes: tuple[str, ...]


def _merge(base: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in child.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge(merged[key], value)
        elif key in merged and isinstance(merged[key], Mapping) != isinstance(value, Mapping):
            raise ConfigError(f"type conflict at {key!r}")
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_experiment(path: str | Path) -> ExperimentConfig:
    seen: set[Path] = set()

    def resolve(candidate: Path) -> tuple[dict[str, Any], tuple[Path, ...], tuple[str, ...]]:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            raise ConfigError(f"configuration inheritance cycle at {candidate}")
        seen.add(candidate)
        try:
            raw = candidate.read_bytes()
            loaded = yaml.safe_load(raw)
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"configuration {candidate} must contain a mapping")
            current = dict(loaded)
            base_ref = current.pop("base_config", None)
            own = sha256(raw).hexdigest()
            if base_ref is None:
                return current, (candidate,), (own,)
            if not isinstance(base_ref, str) or not base_ref or Path(base_ref).is_absolute():
                raise ConfigError("base_config must be a non-empty relative path")
            base, paths, hashes = resolve(candidate.parent / base_ref)
            return _merge(base, current), paths + (candidate,), hashes + (own,)
        except OSError as exc:
            raise ConfigError(f"could not read configuration {candidate}") from exc
        finally:
            seen.remove(candidate)

    data, paths, hashes = resolve(Path(path))
    frozen = _freeze(data)
    return ExperimentConfig(frozen, config_hash(frozen), paths, hashes)


_FULL_REVISION = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _integer(value: Any, path: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ConfigError(f"{path} must be {expected}")


def validate_experiment(config: ExperimentConfig) -> None:
    data = config.data
    model = data.get("model")
    if not isinstance(model, Mapping) or model.get("name") not in MODELS:
        raise ConfigError(f"unknown model: {model.get('name') if isinstance(model, Mapping) else model!r}")
    arm = data.get("allocator")
    if arm not in ALLOCATOR_ARMS:
        raise ConfigError(f"unknown allocator: {arm!r}")
    arms = data.get("arms", ())
    if not isinstance(arms, (list, tuple)) or any(candidate not in ALLOCATOR_ARMS for candidate in arms):
        raise ConfigError("unknown allocator in arms")
    datasets = data.get("datasets", ())
    if not isinstance(datasets, (list, tuple)):
        raise ConfigError("datasets must be a list")
    roles: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, Mapping) or dataset.get("name") not in DATASETS:
            raise ConfigError(f"unknown dataset: {dataset!r}")
        if dataset.get("role") not in DATASET_ROLES:
            raise ConfigError(f"unknown dataset role: {dataset.get('role')!r}")
        roles.add(str(dataset["role"]))
    for baseline in data.get("baselines", ()):
        if not isinstance(baseline, Mapping) or baseline.get("mode") not in PROVENANCE_MODES:
            raise ConfigError(f"unknown provenance mode: {baseline!r}")
        if baseline.get("mode") == "unavailable":
            raise ConfigError(f"baseline is explicitly unavailable: {baseline.get('name')!r}")
    profile = data.get("hardware", "cpu")
    if profile not in HARDWARE_PROFILES:
        raise ConfigError(f"unknown hardware profile: {profile!r}")
    if data.get("formal"):
        if not _FULL_REVISION.fullmatch(str(model.get("revision", ""))):
            raise ConfigError("formal runs require an immutable full model revision")
        for dataset in datasets:
            if not _FULL_REVISION.fullmatch(str(dataset.get("revision", ""))):
                raise ConfigError("formal runs require immutable full dataset revisions")
        if not _DIGEST.fullmatch(str(data.get("container_digest", ""))):
            raise ConfigError("formal runs require a sha256 container digest")
        if tuple(data.get("seeds", ())) != (101, 202, 303):
            raise ConfigError("formal runs require seeds [101, 202, 303]")
        if not DATASET_ROLES.issubset(roles):
            raise ConfigError(f"formal runs require dataset roles {sorted(DATASET_ROLES)}")
        if data.get("stage") != "online":
            raise ConfigError("formal configuration must use online stage")
        if model.get("name") != "qwen25_coder_7b":
            raise ConfigError("formal online runs require the primary model qwen25_coder_7b")
        if profile != "h200_formal":
            raise ConfigError("formal online runs require h200_formal hardware")
        tasks = _mapping(data.get("tasks"), "tasks")
        if not _FULL_REVISION.fullmatch(str(tasks.get("revision", ""))):
            raise ConfigError("formal active task revision must be an immutable full SHA")
        if tasks.get("role") != "train":
            raise ConfigError("formal active tasks must use train role")
        generation = _mapping(data.get("generation"), "generation")
        expected_generation = {"root_samples": 4, "branch_depths": (256, 512, 768), "children_per_branch": 3, "max_new_tokens": 2048}
        for key, expected in expected_generation.items():
            value = tuple(generation.get(key, ())) if key == "branch_depths" else generation.get(key)
            if value != expected:
                raise ConfigError(f"formal generation.{key} must be {expected}")
        _integer(data.get("budget"), "formal budget", 8192)
        training = _mapping(data.get("training"), "training")
        if "updates" in training:
            raise ConfigError("formal training.updates override is forbidden")
        for key, expected in {"warmup_updates": 15, "screening_updates": 200, "formal_updates": 800, "prompt_batch": 64, "group_cap": 8, "optimizer_epochs": 1}.items():
            _integer(training.get(key), f"formal training.{key}", expected)
        prerequisites = _mapping(data.get("prerequisites"), "prerequisites")
        if prerequisites.get("e0_gate_status") != "PASS":
            raise ConfigError("formal E0 gate must be PASS")
        if prerequisites.get("e1_gate_status") != "PASS":
            raise ConfigError("formal E1 gate must be PASS")
        for key in ("e0_artifact_sha256", "e1_artifact_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(prerequisites.get(key, ""))):
                raise ConfigError(f"formal prerequisite {key} must be a SHA-256")
        gradient_audit = _mapping(data.get("gradient_audit"), "gradient_audit")
        if gradient_audit.get("calibration_role") != "calibration":
            raise ConfigError("formal gradient predictor calibration role must be calibration")
        if not re.fullmatch(r"[0-9a-f]{64}", str(gradient_audit.get("predictor_sha256", ""))):
            raise ConfigError("formal gradient predictor must have an immutable SHA-256")
        folds = gradient_audit.get("cross_fit_folds")
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
            raise ConfigError("formal gradient predictor requires cross_fit_folds >= 2")
