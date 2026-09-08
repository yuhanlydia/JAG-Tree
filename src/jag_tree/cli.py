"""Command line entry points that remain offline until execution is requested."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path
import sys
from typing import Sequence

from . import __version__
from .config import ConfigError, canonical_json, load_experiment, thaw, validate_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jag-tree")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report local optional capability availability")
    plan = sub.add_parser("plan", help="validate and hash an experiment without downloads")
    plan.add_argument("config")
    prepare = sub.add_parser("prepare", help="write a create-once canonical experiment plan")
    prepare.add_argument("config")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run", help="launch a staged experiment")
    run.add_argument("config")
    run.add_argument("--output-root", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--backend", choices=("transformers", "fake"), default="transformers")
    run.add_argument("--sandbox", choices=("container", "subprocess", "fake"), default="container")
    run.add_argument("--container-image")
    verify = sub.add_parser("verify", help="verify immutable result artifacts")
    verify.add_argument("path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "doctor":
            print(json.dumps({"jag_tree": __version__, "python": platform.python_version(), "torch": importlib.util.find_spec("torch") is not None, "transformers": importlib.util.find_spec("transformers") is not None}, sort_keys=True))
            return 0
        if args.command == "verify":
            from .artifacts import verify_artifacts
            print(canonical_json(verify_artifacts(args.path)))
            return 0
        config = load_experiment(args.config)
        validate_experiment(config)
        if args.command in {"plan", "prepare"}:
            payload = canonical_json({"config": thaw(config.data), "sha256": config.sha256, "sources": [str(path) for path in config.source_paths]})
            if args.command == "prepare":
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("x", encoding="utf-8") as stream:
                    stream.write(payload + "\n")
                print(output)
            else:
                print(payload)
            return 0
        from .trainer import run_experiment
        backend = sandbox = None
        if not args.dry_run:
            data = config.data
            model = data["model"]
            if args.backend == "fake":
                from .rollout import FakePolicyBackend
                backend = FakePolicyBackend(str(model["revision"]))
            else:
                from .models import TransformersPolicyBackend
                predictor_path = data.get("gradient_audit", {}).get("predictor_path")
                backend = TransformersPolicyBackend(str(model["name"]), str(model["revision"]), str(data.get("hardware", "24gb")), predictor_path)
            if args.sandbox == "fake":
                from .sandbox import FakeSandbox
                sandbox = FakeSandbox()
            elif args.sandbox == "subprocess":
                from .sandbox import SubprocessSandbox
                task_ids = frozenset(str(item) for item in data.get("trusted_fixture_task_ids", ()))
                sandbox = SubprocessSandbox(trusted_task_ids=task_ids)
            else:
                from .sandbox import ContainerSandbox
                if not args.container_image:
                    raise ConfigError("container sandbox requires --container-image with immutable digest")
                sandbox = ContainerSandbox(args.container_image)
        print(canonical_json(run_experiment(config, output_root=args.output_root, seed=args.seed, dry_run=args.dry_run, backend=backend, sandbox=sandbox)))
        return 0
    except (ConfigError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
