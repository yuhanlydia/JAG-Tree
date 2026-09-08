"""Closed registries for reproducible JAG experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    role: str


MODELS = {
    "qwen25_coder_1_5b": ModelSpec("qwen25_coder_1_5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "plumbing"),
    "qwen25_coder_7b": ModelSpec("qwen25_coder_7b", "Qwen/Qwen2.5-Coder-7B-Instruct", "primary"),
    "seed_coder_8b": ModelSpec("seed_coder_8b", "ByteDance-Seed/Seed-Coder-8B-Instruct", "architecture_replication"),
    "deepseek_coder_6_7b": ModelSpec("deepseek_coder_6_7b", "deepseek-ai/deepseek-coder-6.7b-instruct", "cross_family_stress"),
}

DATASETS = frozenset({"taco", "livecodebench_v6", "bigcodebench_full", "bigcodebench_hard", "humaneval_plus", "mbpp_plus", "evaluator_held"})
DATASET_ROLES = frozenset({"train", "dev", "frozen_audit", "public_eval", "locked_final"})
ALLOCATOR_ARMS = frozenset({"uniform", "entropy", "value_variance", "gradient_only", "trace_score", "jag_no_cross", "jag_full", "oracle_moments"})
PROVENANCE_MODES = frozenset({"unavailable", "controlled_ablation"})
HARDWARE_PROFILES = frozenset({"cpu", "16gb", "24gb", "4x24gb", "h200_formal"})
