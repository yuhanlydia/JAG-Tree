from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from jag_tree.models import TransformersPolicyBackend, chat_messages
from jag_tree.rollout import GenerationRequest
from jag_tree.schema import TreeNode
from jag_tree.schema import TaskRecord


def test_16gb_load_uses_transformers_quantization_config(monkeypatch) -> None:
    class QuantizationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert "load_in_4bit" not in kwargs
            assert kwargs["quantization_config"].kwargs == {"load_in_4bit": True}
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=AutoModel,
            AutoTokenizer=AutoTokenizer,
            BitsAndBytesConfig=QuantizationConfig,
        ),
    )
    backend = TransformersPolicyBackend("qwen25_coder_7b", "a" * 40, "16gb")
    backend._load()


def test_24gb_load_uses_bfloat16_without_quantization(monkeypatch) -> None:
    import torch

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert "quantization_config" not in kwargs
            assert kwargs["torch_dtype"] is torch.bfloat16
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=AutoModel,
            AutoTokenizer=AutoTokenizer,
            BitsAndBytesConfig=object,
        ),
    )
    backend = TransformersPolicyBackend("qwen25_coder_7b", "a" * 40, "24gb")
    backend._load()


def test_24gb_lora_does_not_prepare_bfloat16_model_for_kbit(monkeypatch) -> None:
    class Model:
        def eval(self):
            return self

    class AutoTokenizer:
        from_pretrained = staticmethod(lambda *args, **kwargs: object())

    class AutoModel:
        from_pretrained = staticmethod(lambda *args, **kwargs: Model())

    class LoraConfig:
        def __init__(self, **kwargs):
            pass

    def reject_kbit_prepare(model):
        raise AssertionError("BF16 model was incorrectly prepared for k-bit training")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=AutoModel,
            AutoTokenizer=AutoTokenizer,
            BitsAndBytesConfig=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=LoraConfig,
            get_peft_model=lambda model, config: model,
            prepare_model_for_kbit_training=reject_kbit_prepare,
        ),
    )
    backend = TransformersPolicyBackend(
        "qwen25_coder_7b", "a" * 40, "24gb", allow_pilot_predictor=True
    )
    backend._load()


def test_pilot_predictor_is_outcome_blind_and_has_registered_sketch_width() -> None:
    backend = TransformersPolicyBackend(
        "qwen25_coder_7b",
        "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "16gb",
        allow_pilot_predictor=True,
    )
    nodes = (
        TreeNode("root", "task", None, 0, "prompt", (), 0, backend.revision, "v1"),
        TreeNode("edge", "task", "root", 1, "answer", (1,), 1, backend.revision, "v1", reward=1.0, edge_entropy=0.75),
    )
    prediction = backend.frozen_predictor(nodes).predict("edge")
    assert prediction.value_variance == 1.0
    assert prediction.gradient_variance_trace == 1.0
    assert prediction.entropy == 0.75
    np.testing.assert_array_equal(prediction.value_gradient_covariance, np.zeros(256))


def test_pilot_predictor_requires_explicit_opt_in() -> None:
    backend = TransformersPolicyBackend(
        "qwen25_coder_7b",
        "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "16gb",
    )
    node = TreeNode("edge", "task", "root", 1, "answer", (1,), 1, backend.revision, "v1")
    try:
        backend.frozen_predictor((node,))
    except ValueError as exc:
        assert "sealed calibration" in str(exc)
    else:
        raise AssertionError("formal/default backend accepted an unsealed predictor")


def test_generation_seeds_torch_without_passing_unsupported_generator_kwarg() -> None:
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, prompt, return_tensors):
            return Batch(input_ids=torch.tensor([[1, 2]]))

        def decode(self, tokens, skip_special_tokens):
            return "answer"

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            assert "generator" not in kwargs
            return torch.tensor([[1, 2, 3]])

    backend = TransformersPolicyBackend("qwen25_coder_7b", "a" * 40, "16gb")
    backend._model, backend._tokenizer = Model(), Tokenizer()
    text, tokens = backend._generate_text("prompt", 17, GenerationRequest(max_new_tokens=1))
    assert text == "answer" and tokens == (3,)


def test_generation_records_sampled_edge_logprob_and_entropy() -> None:
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, prompt, return_tensors):
            return Batch(input_ids=torch.tensor([[1, 2]]))

        def decode(self, tokens, skip_special_tokens):
            return "answer"

    class Output:
        sequences = torch.tensor([[1, 2, 1]])
        scores = (torch.tensor([[0.0, 2.0]]),)

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return Output()

    backend = TransformersPolicyBackend("qwen25_coder_7b", "a" * 40, "16gb")
    backend._model, backend._tokenizer = Model(), Tokenizer()
    backend._generate_text("prompt", 17, GenerationRequest(max_new_tokens=1))
    logprob, entropy = backend._generation_stats[-1]
    assert logprob < 0
    assert 0 < entropy < 0.5


def test_generation_entropy_ignores_top_p_negative_infinity_logits() -> None:
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, prompt, return_tensors):
            return Batch(input_ids=torch.tensor([[1, 2]]))

        def decode(self, tokens, skip_special_tokens):
            return "answer"

    class Output:
        sequences = torch.tensor([[1, 2, 1]])
        scores = (torch.tensor([[float("-inf"), 2.0]]),)

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return Output()

    backend = TransformersPolicyBackend("qwen25_coder_7b", "a" * 40, "16gb")
    backend._model, backend._tokenizer = Model(), Tokenizer()
    backend._generate_text("prompt", 17, GenerationRequest(max_new_tokens=1))
    assert backend._generation_stats[-1] == (0.0, 0.0)


def test_qwen_code_prompt_requires_python_from_the_first_token() -> None:
    task = TaskRecord("t", "Print the sum.", "", ("",), "fixture", "g", "frozen_audit")
    messages = chat_messages(task, "qwen25_coder_7b")
    assert "first token" in messages[0]["content"].lower()
    assert "no explanation" in messages[0]["content"].lower()
