"""Lazy Hugging Face model adapter and registered chat formatting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import json
import os
import tempfile
from typing import Any

from .registry import HARDWARE_PROFILES, MODELS
from .rollout import GenerationRequest, build_genealogy
from .schema import TaskRecord, TreeNode


def model_plan(name: str, revision: str, hardware: str) -> dict[str, object]:
    if name not in MODELS:
        raise ValueError(f"unknown model: {name}")
    if hardware not in HARDWARE_PROFILES:
        raise ValueError(f"unknown hardware profile: {hardware}")
    model = MODELS[name]
    return {"name": name, "hf_id": model.hf_id, "role": model.role, "revision": revision, "hardware": hardware, "quantization": "nf4" if hardware == "16gb" else "none"}


def chat_messages(task: TaskRecord, model_name: str) -> list[dict[str, str]]:
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    system = (
        "You are an expert competitive programmer. Return only a complete Python solution. "
        "The first token of your response must begin the Python program. No explanation, "
        "analysis, Markdown fence, or prose is allowed."
    )
    if model_name == "seed_coder_8b":
        system = "Solve the programming task. Output only the complete code."
    elif model_name == "deepseek_coder_6_7b":
        system = "Write a correct Python program for the following task."
    return [{"role": "system", "content": system}, {"role": "user", "content": task.statement + (f"\nStarter code:\n{task.starter_code}" if task.starter_code else "")}]


class TransformersPolicyBackend:
    """Sequential generation backend; optional dependencies load on first use."""

    def __init__(self, model_name: str, revision: str, hardware: str = "24gb", calibration_path: str | Path | None = None, *, allow_pilot_predictor: bool = False) -> None:
        self.plan = model_plan(model_name, revision, hardware)
        self.model_name = model_name
        self.revision = revision
        self.hardware = hardware
        self.calibration_path = Path(calibration_path) if calibration_path is not None else None
        self.allow_pilot_predictor = bool(allow_pilot_predictor)
        self._model: Any = None
        self._tokenizer: Any = None
        self._optimizer: Any = None
        self._training_updates = 0
        self._generation_stats: list[tuple[float, float]] = []

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._tokenizer
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Transformers generation requires the optional models extra") from exc
        kwargs: dict[str, Any] = {"revision": self.revision, "device_map": "auto"}
        if self.hardware == "16gb":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self._tokenizer = AutoTokenizer.from_pretrained(self.plan["hf_id"], revision=self.revision)
        self._model = AutoModelForCausalLM.from_pretrained(self.plan["hf_id"], **kwargs)
        if self.allow_pilot_predictor:
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            except ImportError as exc:
                raise RuntimeError("pilot score gradients require PEFT") from exc
            if self.hardware == "16gb":
                self._model = prepare_model_for_kbit_training(self._model)
            self._model = get_peft_model(
                self._model,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj"],
                ),
            )
            self._model.eval()
        return self._model, self._tokenizer

    def _generate_text(self, prompt: str, seed: int, request: GenerationRequest) -> tuple[str, tuple[int, ...]]:
        import torch
        model, tokenizer = self._load()
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        devices = [model.device] if getattr(model.device, "type", None) == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(int(seed))
            if devices:
                torch.cuda.manual_seed_all(int(seed))
            output = model.generate(**encoded, do_sample=True, temperature=request.temperature, top_p=request.top_p, max_new_tokens=request.max_new_tokens, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True)
        sequences = output.sequences if hasattr(output, "sequences") else output
        continuation = sequences[0, encoded["input_ids"].shape[1]:]
        edge_logprob = 0.0
        edge_entropy = 0.0
        scores = tuple(getattr(output, "scores", ()))
        for token, logits in zip(continuation, scores):
            logp = torch.log_softmax(logits[0].float(), dim=-1)
            probability = logp.exp()
            edge_logprob += float(logp[int(token)].item())
            finite = torch.isfinite(logp)
            edge_entropy += float((-(probability[finite] * logp[finite])).sum().item())
        if scores:
            edge_entropy /= len(scores)
        self._generation_stats.append((edge_logprob, edge_entropy))
        return tokenizer.decode(continuation, skip_special_tokens=True), tuple(int(token) for token in continuation.detach().cpu().tolist())

    def generate_tree(self, task: TaskRecord, seed: int, request: GenerationRequest | None = None) -> tuple[TreeNode, ...]:
        request = request or GenerationRequest()
        model, tokenizer = self._load()
        _ = model
        messages = chat_messages(task, self.model_name)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        def sample_edge(prefix: tuple[int, ...], count: int, edge_seed: int) -> tuple[int, ...]:
            conditional_prompt = prompt + tokenizer.decode(prefix, skip_special_tokens=True)
            _, tokens = self._generate_text(conditional_prompt, edge_seed, replace(request, max_new_tokens=count, branch_depths=()))
            return tokens

        self._generation_stats = []
        nodes = build_genealogy(task, prompt, self.revision, f"{self.model_name}-chat-v1", seed, request, sample_edge, lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True), lambda edge: bool(edge and edge[-1] == tokenizer.eos_token_id))
        stats = iter(self._generation_stats)
        return tuple(
            node if node.parent_id is None else replace(node, edge_logprob=(stat := next(stats))[0], edge_entropy=stat[1])
            for node in nodes
        )

    def score_tree(self, task: TaskRecord, nodes: tuple[TreeNode, ...]) -> dict[str, Any]:
        """Sketch actual sampled-edge log-probability gradients."""

        import torch
        from .gradient import GradientSpec, ScoreExample, score_sketch

        model, tokenizer = self._load()
        root = next(node for node in nodes if node.parent_id is None)
        by_id = {node.node_id: node for node in nodes}
        examples = []
        edge_ids = []
        for node in nodes:
            if node.parent_id is None:
                continue
            ancestors = []
            current = node
            while current.parent_id is not None:
                ancestors.append(current)
                current = by_id[current.parent_id]
            ancestors.reverse()
            prefix = tuple(token for edge in ancestors[:-1] for token in edge.token_ids)
            prompt_ids = tuple(tokenizer(root.text, add_special_tokens=False)["input_ids"])
            input_ids = torch.tensor([prompt_ids + prefix + node.token_ids], device=model.device, dtype=torch.long)
            examples.append(ScoreExample({"input_ids": input_ids}, len(prompt_ids) + len(prefix), len(node.token_ids)))
            edge_ids.append(node.node_id)
        registered = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not registered:
            raise ValueError("policy has no registered differentiable parameters")
        preferred = [name for name, _ in registered if "lora" in name.lower()]
        # One fixed late adapter tensor keeps the pilot sketch auditable and small
        # enough for a 16GB worker. Formal configs use their sealed block list.
        names = tuple(preferred[-1:] if self.allow_pilot_predictor and preferred else preferred or [registered[-1][0]])
        result = score_sketch(model, examples, GradientSpec(names, dimension=256, seed=0))
        return {node_id: result.sketch[index] for index, node_id in enumerate(edge_ids)}

    def frozen_predictor(self, nodes: tuple[TreeNode, ...]) -> object:
        """Load a sealed cross-fit calibration artifact; never fit on audit rows."""

        from .audit import FrozenMomentPredictor, MomentPrediction

        if self.calibration_path is None and self.allow_pilot_predictor:
            import numpy as np

            predictions = {
                node.node_id: MomentPrediction(
                    1.0,
                    1.0,
                    np.zeros(256, dtype=np.float64),
                    max(0.0, float(node.edge_entropy)),
                    1.0,
                )
                for node in nodes
                if node.parent_id is not None
            }
            return FrozenMomentPredictor("pilot:outcome-blind-structural-v1", predictions)
        del nodes
        if self.calibration_path is None or not self.calibration_path.is_file():
            raise ValueError("transformers audit requires a sealed calibration predictor artifact")
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        if payload.get("policy_revision") != self.revision or payload.get("role") != "calibration":
            raise ValueError("calibration predictor identity/role does not match policy")
        predictions = {
            node_id: MomentPrediction(
                float(row["value_variance"]),
                float(row["gradient_variance_trace"]),
                row["value_gradient_covariance"],
                float(row["entropy"]),
                float(row["trace_score"]),
            )
            for node_id, row in payload["predictions"].items()
        }
        return FrozenMomentPredictor(sha256(self.calibration_path.read_bytes()).hexdigest(), predictions)

    def _conditional_edge_logprobs(self, bank: Any, model: Any, tokenizer: Any) -> dict[str, Any]:
        """Differentiate only each stored edge under its original prompt/prefix."""

        import torch
        from .gradient import causal_edge_logprob

        by_id = {node.node_id: node for node in bank.nodes}
        roots = {node.task_id: node for node in bank.nodes if node.parent_id is None}
        results: dict[str, Any] = {}
        for node in bank.nodes:
            if node.parent_id is None:
                continue
            ancestors: list[Any] = []
            current = node
            while current.parent_id is not None:
                ancestors.append(current)
                current = by_id[current.parent_id]
            ancestors.reverse()
            prefix_tokens = tuple(token for edge in ancestors[:-1] for token in edge.token_ids)
            prompt_ids = tuple(tokenizer(roots[node.task_id].text, add_special_tokens=False)["input_ids"])
            input_ids = torch.tensor([prompt_ids + prefix_tokens + node.token_ids], device=model.device, dtype=torch.long)
            edge_start = len(prompt_ids) + len(prefix_tokens)
            if edge_start < 1 or not node.token_ids:
                raise ValueError(f"edge {node.node_id} has no causal prompt/token support")
            logits = model(input_ids=input_ids).logits
            results[node.node_id] = causal_edge_logprob(logits, input_ids, edge_start, len(node.token_ids))
        return results

    def prepare_training_update(self, bank_path: Any, *, replay_rows: tuple[dict[str, object], ...]) -> tuple[Any, Any, Any, Any]:
        """Build one differentiable unique-edge QLoRA update from a sealed bank."""

        try:
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("online training requires the optional training extra") from exc
        from .bank import verify_bank
        from .ledger import RunLedger
        from .trainer import QLORA_PROFILES, TrainingBatch, replay_edge_coefficients

        if self.hardware not in QLORA_PROFILES:
            raise ValueError(f"online QLoRA does not support hardware profile {self.hardware!r}")
        model, tokenizer = self._load()
        profile = QLORA_PROFILES[self.hardware]
        if self._optimizer is None:
            if profile.quantized:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, LoraConfig(r=profile.rank, lora_alpha=profile.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
            self._model = model
            self._optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-6)
        bank = verify_bank(bank_path)
        edge_logprobs = self._conditional_edge_logprobs(bank, model, tokenizer)
        coefficients = replay_edge_coefficients(bank, replay_rows)
        selected = set(coefficients)
        if not selected or not selected.issubset(edge_logprobs):
            raise ValueError("replay paths contain unknown edges")
        edge_ids = tuple(sorted(selected))
        logprobs = torch.stack([edge_logprobs[node_id] for node_id in edge_ids])
        credit = torch.as_tensor([coefficients[node_id] for node_id in edge_ids], device=logprobs.device, dtype=logprobs.dtype)
        edge_mask = torch.eye(len(edge_ids), device=logprobs.device, dtype=torch.bool)
        weights = torch.ones_like(credit)
        ledger = RunLedger(generation_tokens=sum(node.unique_tokens for node in bank.nodes if node.node_id in selected), optimizer_steps=1)
        return model, self._optimizer, TrainingBatch(logprobs, edge_mask, credit, weights), ledger

    def train_bank_one_update(self, bank_path: Any, *, replay_rows: tuple[dict[str, object], ...], expected_ledger: Any) -> float:
        from .trainer import train_one_update
        from .bank import verify_bank
        from .ledger import RunLedger

        model, optimizer, batch, prepared_expected = self.prepare_training_update(bank_path, replay_rows=replay_rows)
        if not prepared_expected.compare(expected_ledger, token_tolerance=0.0, cpu_tolerance=0.0).matched:
            raise ValueError("selected-edge ledger differs from runner budget")
        loss = train_one_update(model, optimizer, batch)
        measured = RunLedger()
        selected_edge_ids = {str(edge_id) for row in replay_rows for edge_id in row["path_edge_ids"]}
        for node in verify_bank(bank_path).nodes:
            if node.node_id in selected_edge_ids:
                measured.record_generation(node.unique_tokens)
        measured.record_optimization(gpu_hours=0.0, steps=1)
        comparison = expected_ledger.compare(measured, token_tolerance=0.01, cpu_tolerance=0.05)
        if not comparison.matched:
            raise ValueError(f"measured training ledger differs from registered budget: {comparison}")
        self._training_updates += 1
        checkpoint = Path(bank_path).parent / f"checkpoint-{self._training_updates:04d}"
        if checkpoint.exists():
            raise FileExistsError(f"checkpoint already exists: {checkpoint}")
        staging = Path(tempfile.mkdtemp(prefix=f".{checkpoint.name}-", dir=checkpoint.parent))
        try:
            model.save_pretrained(staging)
            import torch
            torch.save({"optimizer": optimizer.state_dict(), "updates": self._training_updates, "revision": self.revision}, staging / "optimizer.pt")
            os.replace(staging, checkpoint)
        except Exception:
            import shutil
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return loss
