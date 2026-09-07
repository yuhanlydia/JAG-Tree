"""Configuration inheritance and reproducible configuration identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import copy
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedConfig:
    """A fully resolved configuration together with its provenance."""

    _data: Mapping[str, Any]
    sha256: str
    source_paths: tuple[Path, ...]
    source_hashes: tuple[str, ...]

    @property
    def data(self) -> dict[str, Any]:
        """Return a defensive mutable copy of the hashed resolved mapping."""

        return _thaw(self._data)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _canonical_json(data: Any) -> str:
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError("configuration must be JSON-canonicalizable") from exc


def config_hash(data: Any) -> str:
    """Return the canonical SHA-256 identity of JSON-compatible data."""

    return sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def deep_merge(base: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively while treating lists and nulls as replacements."""

    if not isinstance(base, Mapping) or not isinstance(child, Mapping):
        raise ConfigError("deep_merge requires two mappings")
    merged = copy.deepcopy(dict(base))
    for key, child_value in child.items():
        if key not in merged:
            merged[key] = copy.deepcopy(child_value)
            continue
        base_value = merged[key]
        if child_value is None:
            merged[key] = None
        elif isinstance(base_value, Mapping) and isinstance(child_value, Mapping):
            merged[key] = deep_merge(base_value, child_value)
        elif isinstance(base_value, Mapping) != isinstance(child_value, Mapping):
            raise ConfigError(f"type conflict at {key!r}: mappings cannot replace non-mappings")
        else:
            merged[key] = copy.deepcopy(child_value)
    return merged


def _read_mapping(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"could not read configuration {path}") from exc
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}") from exc
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"configuration {path} must contain a mapping")
    return dict(loaded), sha256(raw).hexdigest()


def load_config(path: str | Path) -> ResolvedConfig:
    """Resolve a YAML configuration and its relative ``base_config`` ancestry."""

    seen: set[Path] = set()

    def resolve(candidate: Path) -> tuple[dict[str, Any], tuple[Path, ...], tuple[str, ...]]:
        canonical_path = candidate.expanduser().resolve()
        if canonical_path in seen:
            raise ConfigError(f"configuration inheritance cycle at {canonical_path}")
        seen.add(canonical_path)
        try:
            current, current_hash = _read_mapping(canonical_path)
            base_ref = current.pop("base_config", None)
            if base_ref is None:
                return current, (canonical_path,), (current_hash,)
            if not isinstance(base_ref, str) or not base_ref:
                raise ConfigError(f"base_config in {canonical_path} must be a non-empty relative path")
            base_path = Path(base_ref)
            if base_path.is_absolute():
                raise ConfigError(f"base_config in {canonical_path} must be relative")
            base_data, base_paths, base_hashes = resolve(canonical_path.parent / base_path)
            return deep_merge(base_data, current), base_paths + (canonical_path,), base_hashes + (current_hash,)
        finally:
            seen.remove(canonical_path)

    data, source_paths, source_hashes = resolve(Path(path))
    return ResolvedConfig(_freeze(data), config_hash(data), source_paths, source_hashes)
