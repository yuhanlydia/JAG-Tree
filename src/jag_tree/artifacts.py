"""Atomic immutable result artifacts and checksum verification."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping


class ArtifactError(ValueError):
    """Artifact set is incomplete, malformed, or has been modified."""


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_artifacts(path: str | Path, result: Mapping[str, object], *, rows: Iterable[Mapping[str, object]], manifest: Mapping[str, object] | None = None) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"result path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        (staging / "result.json").write_text(_json(dict(result)) + "\n", encoding="utf-8")
        (staging / "rows.jsonl").write_text("".join(_json(dict(row)) + "\n" for row in rows), encoding="utf-8")
        (staging / "manifest.json").write_text(_json(dict(manifest or {})) + "\n", encoding="utf-8")
        checksums = {name: _digest(staging / name) for name in ("result.json", "rows.jsonl", "manifest.json")}
        (staging / "checksums.json").write_text(_json(checksums) + "\n", encoding="utf-8")
        (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def verify_artifacts(path: str | Path) -> dict[str, object]:
    root = Path(path)
    try:
        if not (root / "COMPLETE").is_file():
            raise ArtifactError("artifact directory is incomplete")
        checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
        if set(checksums) != {"result.json", "rows.jsonl", "manifest.json"}:
            raise ArtifactError("artifact checksum inventory must exactly match payloads")
        for name, digest in checksums.items():
            if _digest(root / name) != digest:
                raise ArtifactError(f"checksum mismatch for {name}")
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        if result.get("status") not in {"PASS", "FAIL", "INCOMPLETE", "INVALID"}:
            raise ArtifactError("unknown result status")
        if result.get("status") == "PASS":
            raise ArtifactError("scientific PASS verification is unsupported; formal evidence remains INCOMPLETE")
        return result
    except ArtifactError:
        raise
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ArtifactError(f"invalid artifacts: {exc}") from exc
