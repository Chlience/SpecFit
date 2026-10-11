#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from specfit.runtime import Eagle3ChainDrafter, Eagle3TreeDrafter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    model_lock = json.loads(
        Path("/opt/specfit/models.lock.json").read_text(encoding="utf-8")
    )
    target_path = args.model_root / model_lock["target"]["directory"]
    draft_path = args.model_root / model_lock["drafter"]["directory"]

    target_config = AutoConfig.from_pretrained(target_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)

    adapter = Eagle3ChainDrafter.from_pretrained(
        draft_path,
        target_root=target_path,
        device="cuda" if args.gpu else "cpu",
        use_torch_compile=False,
        expected_target_repo_id=model_lock["target"]["repo_id"],
    )
    drafter = adapter.model

    result: dict[str, object] = {
        "target_config": type(target_config).__name__,
        "target_layers": target_config.num_hidden_layers,
        "target_experts": target_config.num_experts,
        "target_top_k": target_config.num_experts_per_tok,
        "tokenizer_size": len(tokenizer),
        "drafter_class": type(drafter).__name__,
        "drafter_parameters": sum(parameter.numel() for parameter in drafter.parameters()),
        "drafter_dtype": str(next(drafter.parameters()).dtype),
        "verifier_path": drafter.config.speculators_config.verifier.name_or_path,
        "draft_vocabulary_mode": adapter.vocabulary_mode,
        "draft_vocabulary_size": adapter.draft_vocab_size,
        "target_vocabulary_size": adapter.target_vocab_size,
    }

    if args.gpu:
        hidden_states = torch.randn(
            1, 4, 3 * target_config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        fed_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
        pending_seed = torch.tensor([[5]], device="cuda")
        with torch.inference_mode():
            chain_tokens = adapter.propose(
                fed_ids=fed_ids,
                aux_history=hidden_states,
                pending_seed=pending_seed,
                draft_length=3,
            )
            tree = Eagle3TreeDrafter(adapter).propose_tree(
                fed_ids=fed_ids,
                aux_history=hidden_states,
                pending_seed=pending_seed,
                top_k=2,
                max_depth=3,
                draft_node_budget=7,
            )
        root_node_ids = tuple(
            node_id
            for node_id, parent_id in enumerate(tree.parent_ids, start=1)
            if parent_id == 0
        )
        highest_probability_root = min(
            root_node_ids,
            key=lambda node_id: (
                -tree.conditional_probabilities[node_id - 1],
                tree.token_ids[node_id - 1],
            ),
        )
        if (
            tree.token_ids[highest_probability_root - 1]
            != int(chain_tokens[0, 0].item())
        ):
            raise RuntimeError(
                "chain Top-1 differs from the shallow tree root Top-1"
            )
        result["synthetic_forward"] = {
            "chain_token_ids": chain_tokens[0].tolist(),
            "tree_token_ids": tree.token_ids,
            "tree_parent_ids": tree.parent_ids,
            "tree_depths": tree.depths,
            "tree_conditional_probabilities": tree.conditional_probabilities,
            "tree_cumulative_log_probabilities": (
                tree.cumulative_log_probabilities
            ),
            "first_greedy_token_preserved": True,
        }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
