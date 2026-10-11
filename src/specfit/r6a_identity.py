
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .r6a_experiment import _int, _map, _sha

R6A_DATASET_IDENTITY = {
    "repo_id": "RedHatAI/speculator_benchmarks",
    "revision": "e383d21cf8c005b0fcba4b8d27a2800a075fed0d",
    "filename": "math_reasoning.jsonl",
    "category": "math_reasoning",
}
R6A_MODEL_IDENTITIES = {
    "target": {
        "repo_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
    },
    "drafter": {
        "repo_id": (
            "RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
        ),
        "revision": "a7600ef6ca94c4e06cc1022879944be15949aee4",
    },
}

@dataclass(frozen=True, slots=True)
class ExperimentIdentity:

    question_id: int
    prompt_input_ids_length: int
    prompt_input_ids_sha256: str

    @classmethod
    def read(
        cls,
        root: Mapping[str, object],
        config: Mapping[str, object],
    ) -> ExperimentIdentity:
        dataset = _map("dataset", root.get("dataset"))
        if any(
            dataset.get(field) != expected
            for field, expected in R6A_DATASET_IDENTITY.items()
        ):
            raise ValueError("dataset identity does not match the GSM8K lock")
        question_id = _int("dataset.question_id", dataset.get("question_id"), 1)
        if question_id != 480:
            raise ValueError("R6a-TB1 dataset question_id must equal 480")

        prompt = _map("prompt", root.get("prompt"))
        input_length = _int(
            "prompt.input_ids_length",
            prompt.get("input_ids_length"),
            1,
        )
        input_digest = _sha(
            "prompt.input_ids_sha256", prompt.get("input_ids_sha256")
        )

        models = _map("models", root.get("models"))
        for model_name, expected_identity in R6A_MODEL_IDENTITIES.items():
            model = _map(f"models.{model_name}", models.get(model_name))
            if any(
                model.get(field) != expected
                for field, expected in expected_identity.items()
            ):
                raise ValueError(f"models.{model_name} does not match model lock")
        if config.get("dtype") != "bfloat16":
            raise ValueError("config.dtype must equal bfloat16")
        if config.get("numeric_mode") != "math_bf16_full_reduction":
            raise ValueError(
                "config.numeric_mode must equal math_bf16_full_reduction"
            )
        return cls(question_id, input_length, input_digest)


__all__ = [
    "ExperimentIdentity",
    "R6A_DATASET_IDENTITY",
    "R6A_MODEL_IDENTITIES",
]
