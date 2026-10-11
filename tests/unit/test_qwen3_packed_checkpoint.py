from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import specfit.adapters.qwen3_packed_checkpoint as packed_module
from specfit.adapters.qwen3_checkpoint import CheckpointLayout
from specfit.adapters.qwen3_packed_checkpoint import (
    PackedCheckpointError,
    PackedQwen3Checkpoint,
    pack_qwen3_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Qwen3PackedCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        self.output = self.root / "packed"
        self.source.mkdir()
        self.source_tensors = self._write_source_checkpoint(self.source)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_source_checkpoint(root: Path) -> dict[str, torch.Tensor]:
        config = {
            "model_type": "qwen3_moe",
            "torch_dtype": "float32",
            "num_hidden_layers": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "hidden_size": 2,
            "moe_intermediate_size": 3,
        }
        (root / "config.json").write_text(
            json.dumps(config, sort_keys=True),
            encoding="utf-8",
        )

        tensors: dict[str, torch.Tensor] = {}
        next_value = 1
        for layer_id in range(2):
            for expert_id in range(2):
                prefix = (
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}"
                )
                for projection, shape in (
                    ("gate_proj", (3, 2)),
                    ("up_proj", (3, 2)),
                    ("down_proj", (2, 3)),
                ):
                    tensors[f"{prefix}.{projection}.weight"] = torch.arange(
                        next_value,
                        next_value + 6,
                        dtype=torch.float32,
                    ).reshape(shape)
                    next_value += 6
        tensors["model.embed_tokens.weight"] = torch.arange(
            8,
            dtype=torch.float32,
        ).reshape(4, 2)
        tensors["model.norm.weight"] = torch.tensor(
            [0.25, 0.5],
            dtype=torch.float32,
        )

        names = tuple(tensors)
        first_names = names[::2]
        second_names = names[1::2]
        shard_names = (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )
        save_file(
            {name: tensors[name] for name in first_names},
            root / shard_names[0],
        )
        save_file(
            {name: tensors[name] for name in second_names},
            root / shard_names[1],
        )
        weight_map = {
            name: shard_names[index % 2]
            for index, name in enumerate(names)
        }
        index = {
            "metadata": {
                "total_size": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in tensors.values()
                )
            },
            "weight_map": weight_map,
        }
        (root / "model.safetensors.index.json").write_text(
            json.dumps(index, sort_keys=True),
            encoding="utf-8",
        )
        return tensors

    def test_pack_preserves_expert_layout_and_nonexpert_tensors(
        self,
    ) -> None:
        source_layout = CheckpointLayout.from_pretrained(self.source)

        packed = pack_qwen3_checkpoint(
            self.source,
            self.output,
            source_repo_id="fixture/Qwen3-MoE",
            source_revision="fixture-revision",
        )

        reopened = PackedQwen3Checkpoint.open(
            self.output,
            source_layout=source_layout,
            verify_hashes=True,
        )
        self.assertEqual(packed, reopened)
        self.assertEqual(reopened.census.num_hidden_layers, 2)
        self.assertEqual(reopened.census.num_experts, 2)
        self.assertEqual(reopened.packed_expert_elements, 18)
        self.assertTrue(reopened.manifest["complete"])

        for layer_id in range(2):
            with safe_open(
                reopened.layer_path(layer_id),
                framework="pt",
                device="cpu",
            ) as handle:
                self.assertEqual(tuple(handle.keys()), ("experts",))
                actual = handle.get_tensor("experts")
            expected_rows = []
            for expert_id in range(2):
                prefix = (
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}"
                )
                expected_rows.append(
                    torch.cat(
                        tuple(
                            self.source_tensors[
                                f"{prefix}.{projection}.weight"
                            ].reshape(-1)
                            for projection in (
                                "gate_proj",
                                "up_proj",
                                "down_proj",
                            )
                        )
                    )
                )
            torch.testing.assert_close(actual, torch.stack(expected_rows))

        with safe_open(
            reopened.nonexpert_path,
            framework="pt",
            device="cpu",
        ) as handle:
            self.assertEqual(
                set(handle.keys()),
                {"model.embed_tokens.weight", "model.norm.weight"},
            )
            for name in handle.keys():
                torch.testing.assert_close(
                    handle.get_tensor(name),
                    self.source_tensors[name],
                )

        manifest = reopened.manifest
        self.assertEqual(
            manifest["source"]["config_sha256"],
            _sha256(self.source / "config.json"),
        )
        self.assertEqual(
            manifest["source"]["index_sha256"],
            _sha256(self.source / "model.safetensors.index.json"),
        )
        self.assertEqual(
            manifest["source"]["repo_id"],
            "fixture/Qwen3-MoE",
        )
        self.assertEqual(
            manifest["source"]["revision"],
            "fixture-revision",
        )
        records = [manifest["files"]["nonexperts"]]
        records.extend(manifest["files"]["layers"])
        for record in records:
            path = self.output / record["filename"]
            self.assertEqual(record["size_bytes"], path.stat().st_size)
            self.assertEqual(record["sha256"], _sha256(path))

    def test_source_tensors_are_read_shard_first_in_physical_order(
        self,
    ) -> None:
        index = json.loads(
            (self.source / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        expected = []
        for shard_name in sorted(set(index["weight_map"].values())):
            expected.extend(
                packed_module._tensor_names_in_storage_order(
                    self.source / shard_name
                )
            )

        observed = []
        original = packed_module._read_source_tensor

        def recording_read(handle: object, name: str) -> torch.Tensor:
            observed.append(name)
            return original(handle, name)

        with mock.patch.object(
            packed_module,
            "_read_source_tensor",
            side_effect=recording_read,
        ):
            pack_qwen3_checkpoint(self.source, self.output)

        self.assertEqual(observed, expected)

    def test_open_rejects_missing_or_incomplete_manifest(self) -> None:
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        with self.assertRaisesRegex(PackedCheckpointError, "manifest"):
            PackedQwen3Checkpoint.open(incomplete)

        (incomplete / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": packed_module.PACKED_CHECKPOINT_SCHEMA,
                    "schema_version": packed_module.PACKED_CHECKPOINT_VERSION,
                    "complete": False,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(PackedCheckpointError, "complete"):
            PackedQwen3Checkpoint.open(incomplete)

    def test_open_rejects_source_hash_mismatch(self) -> None:
        pack_qwen3_checkpoint(self.source, self.output)
        source_layout = CheckpointLayout.from_pretrained(self.source)
        config_path = self.source / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["fixture_revision"] = 2
        config_path.write_text(
            json.dumps(config, sort_keys=True),
            encoding="utf-8",
        )
        changed_layout = CheckpointLayout.from_pretrained(self.source)
        self.assertNotEqual(
            source_layout.config_sha256,
            changed_layout.config_sha256,
        )

        with self.assertRaisesRegex(PackedCheckpointError, "config SHA256"):
            PackedQwen3Checkpoint.open(
                self.output,
                source_layout=changed_layout,
            )

    def test_open_validates_source_metadata_without_weight_shards(
        self,
    ) -> None:
        pack_qwen3_checkpoint(self.source, self.output)
        for shard in self.source.glob("model-*.safetensors"):
            shard.unlink()

        reopened = PackedQwen3Checkpoint.open(
            self.output,
            source_metadata_root=self.source,
        )

        self.assertEqual(reopened.census.num_hidden_layers, 2)
        self.assertEqual(reopened.census.num_experts, 2)

    def test_source_metadata_validation_rejects_hash_mismatch(self) -> None:
        pack_qwen3_checkpoint(self.source, self.output)
        config_path = self.source / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["fixture_revision"] = 3
        config_path.write_text(
            json.dumps(config, sort_keys=True),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(PackedCheckpointError, "config SHA256"):
            PackedQwen3Checkpoint.open(
                self.output,
                source_metadata_root=self.source,
            )

    def test_explicit_hash_verification_detects_same_size_corruption(
        self,
    ) -> None:
        packed = pack_qwen3_checkpoint(self.source, self.output)
        layer_path = packed.layer_path(0)
        with layer_path.open("r+b") as handle:
            handle.seek(-1, 2)
            original = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes((original[0] ^ 0xFF,)))

        PackedQwen3Checkpoint.open(self.output, verify_hashes=False)
        with self.assertRaisesRegex(PackedCheckpointError, "SHA256"):
            PackedQwen3Checkpoint.open(
                self.output,
                verify_hashes=True,
            )

    def test_pack_refuses_existing_destination(self) -> None:
        self.output.mkdir()

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            pack_qwen3_checkpoint(self.source, self.output)


if __name__ == "__main__":
    unittest.main()
