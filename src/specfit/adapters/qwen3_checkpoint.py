
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from math import prod
from pathlib import Path

from specfit.state import ExpertKey

_EXPERT_KEY = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TensorLocation:
    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True, slots=True)
class ExpertSpec:
    key: ExpertKey
    gate_proj: TensorLocation
    up_proj: TensorLocation
    down_proj: TensorLocation

    @property
    def nbytes(self) -> int:
        return (
            self.gate_proj.nbytes
            + self.up_proj.nbytes
            + self.down_proj.nbytes
        )


@dataclass(frozen=True, slots=True)
class ModelCensus:
    model_type: str
    torch_dtype: str
    num_hidden_layers: int
    num_experts: int
    num_experts_per_tok: int
    hidden_size: int
    moe_intermediate_size: int
    shared_expert_tensor_count: int
    checkpoint_total_bytes: int
    routed_expert_bytes: int
    nonexpert_bytes: int
    expert_bytes: int
    expert_count: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "model_type": self.model_type,
            "torch_dtype": self.torch_dtype,
            "num_hidden_layers": self.num_hidden_layers,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_tensor_count": self.shared_expert_tensor_count,
            "checkpoint_total_bytes": self.checkpoint_total_bytes,
            "routed_expert_bytes": self.routed_expert_bytes,
            "nonexpert_bytes": self.nonexpert_bytes,
            "expert_bytes": self.expert_bytes,
            "expert_count": self.expert_count,
        }


@dataclass(frozen=True, slots=True)
class CheckpointLayout:
    model_root: Path
    config_sha256: str
    index_sha256: str
    census: ModelCensus
    experts: tuple[ExpertSpec, ...]

    @classmethod
    def from_pretrained(cls, model_root: str | Path) -> CheckpointLayout:
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError(
                "safetensors is required to inspect checkpoint tensor headers"
            ) from error

        root = Path(model_root).resolve()
        config_path = root / "config.json"
        index_path = root / "model.safetensors.index.json"
        config = json.loads(config_path.read_text())
        index = json.loads(index_path.read_text())
        weight_map: dict[str, str] = index["weight_map"]

        grouped_names: dict[str, list[str]] = {}
        parsed_keys: dict[str, tuple[int, int, str]] = {}
        for name, shard in weight_map.items():
            match = _EXPERT_KEY.match(name)
            if match is None:
                continue
            parsed_keys[name] = (
                int(match.group("layer")),
                int(match.group("expert")),
                match.group("projection"),
            )
            grouped_names.setdefault(shard, []).append(name)

        tensors: dict[tuple[int, int], dict[str, TensorLocation]] = {}
        for shard, names in sorted(grouped_names.items()):
            shard_path = root / shard
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for name in names:
                    layer_id, expert_id, projection = parsed_keys[name]
                    view = handle.get_slice(name)
                    shape = tuple(view.get_shape())
                    dtype = view.get_dtype()
                    try:
                        element_bytes = _DTYPE_BYTES[dtype]
                    except KeyError as error:
                        raise ValueError(
                            f"unsupported tensor dtype {dtype!r} for {name}"
                        ) from error
                    tensors.setdefault((layer_id, expert_id), {})[projection] = (
                        TensorLocation(
                            name=name,
                            shard=shard,
                            shape=shape,
                            dtype=dtype,
                            nbytes=prod(shape) * element_bytes,
                        )
                    )

        num_layers = int(config["num_hidden_layers"])
        num_experts = int(
            config.get("num_experts") or config["num_local_experts"]
        )
        hidden_size = int(config["hidden_size"])
        intermediate_size = int(config["moe_intermediate_size"])
        expected_experts = num_layers * num_experts
        if len(tensors) != expected_experts:
            raise ValueError(
                f"found {len(tensors)} routed experts, expected {expected_experts}"
            )

        experts = []
        for layer_id in range(num_layers):
            for expert_id in range(num_experts):
                projections = tensors[(layer_id, expert_id)]
                if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
                    raise ValueError(
                        f"expert {(layer_id, expert_id)} has projections "
                        f"{sorted(projections)}"
                    )
                expected_shapes = {
                    "gate_proj": (intermediate_size, hidden_size),
                    "up_proj": (intermediate_size, hidden_size),
                    "down_proj": (hidden_size, intermediate_size),
                }
                for projection, expected_shape in expected_shapes.items():
                    if projections[projection].shape != expected_shape:
                        raise ValueError(
                            f"{projections[projection].name} has shape "
                            f"{projections[projection].shape}, "
                            f"expected {expected_shape}"
                        )
                experts.append(
                    ExpertSpec(
                        key=ExpertKey(layer_id, expert_id),
                        gate_proj=projections["gate_proj"],
                        up_proj=projections["up_proj"],
                        down_proj=projections["down_proj"],
                    )
                )

        expert_sizes = {expert.nbytes for expert in experts}
        if len(expert_sizes) != 1:
            raise ValueError(f"expert sizes are not uniform: {expert_sizes}")
        routed_expert_bytes = sum(expert.nbytes for expert in experts)
        checkpoint_total_bytes = int(index["metadata"]["total_size"])
        if routed_expert_bytes > checkpoint_total_bytes:
            raise ValueError("routed expert bytes exceed checkpoint total size")

        shared_expert_tensor_count = sum(
            ".shared_expert." in name for name in weight_map
        )
        census = ModelCensus(
            model_type=str(config["model_type"]),
            torch_dtype=str(
                config.get("dtype") or config.get("torch_dtype")
            ),
            num_hidden_layers=num_layers,
            num_experts=num_experts,
            num_experts_per_tok=int(config["num_experts_per_tok"]),
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            shared_expert_tensor_count=shared_expert_tensor_count,
            checkpoint_total_bytes=checkpoint_total_bytes,
            routed_expert_bytes=routed_expert_bytes,
            nonexpert_bytes=checkpoint_total_bytes - routed_expert_bytes,
            expert_bytes=next(iter(expert_sizes)),
            expert_count=len(experts),
        )
        return cls(
            model_root=root,
            config_sha256=_sha256(config_path),
            index_sha256=_sha256(index_path),
            census=census,
            experts=tuple(experts),
        )

    def expert(self, key: ExpertKey) -> ExpertSpec:
        index = key.layer_id * self.census.num_experts + key.expert_id
        try:
            expert = self.experts[index]
        except IndexError as error:
            raise KeyError(key) from error
        if expert.key != key:
            raise KeyError(key)
        return expert

    def expert_cache_bytes(self, slots_per_layer: int) -> int:
        if not 0 <= slots_per_layer <= self.census.num_experts:
            raise ValueError(
                f"slots_per_layer must be in [0, {self.census.num_experts}]"
            )
        return (
            slots_per_layer
            * self.census.num_hidden_layers
            * self.census.expert_bytes
        )

    def summary(self, slot_counts: tuple[int, ...]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_root": str(self.model_root),
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
            "census": self.census.as_dict(),
            "expert_cache": [
                {
                    "slots_per_layer": slots,
                    "resident_fraction": slots / self.census.num_experts,
                    "bytes": self.expert_cache_bytes(slots),
                    "gib": self.expert_cache_bytes(slots) / 2**30,
                }
                for slots in slot_counts
            ],
        }
