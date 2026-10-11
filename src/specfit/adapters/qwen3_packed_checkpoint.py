
from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .qwen3_checkpoint import CheckpointLayout, ModelCensus

PACKED_CHECKPOINT_SCHEMA = "specfit.qwen3_packed_checkpoint"
PACKED_CHECKPOINT_VERSION = 1
PACKED_MANIFEST_FILENAME = "manifest.json"
PACKED_EXPERT_LAYOUT = ("gate_proj", "up_proj", "down_proj")
_LOGGER = logging.getLogger(__name__)

_TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}


class PackedCheckpointError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        encoded_length = handle.read(8)
        if len(encoded_length) != 8:
            raise PackedCheckpointError(
                f"safetensors header length is missing: {path}"
            )
        header_length = struct.unpack("<Q", encoded_length)[0]
        if header_length <= 0 or header_length > file_size - 8:
            raise PackedCheckpointError(
                f"invalid safetensors header length in {path}"
            )
        encoded_header = handle.read(header_length)
    try:
        raw_header = json.loads(encoded_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackedCheckpointError(
            f"invalid safetensors header JSON in {path}"
        ) from error
    if not isinstance(raw_header, dict):
        raise PackedCheckpointError(
            f"safetensors header must be an object: {path}"
        )

    header: dict[str, dict[str, Any]] = {}
    for name, entry in raw_header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise PackedCheckpointError(
                f"invalid tensor header entry in {path}"
            )
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise PackedCheckpointError(
                f"invalid data_offsets for {name} in {path}"
            )
        header[name] = entry
    return header


def _tensor_names_in_storage_order(path: Path) -> tuple[str, ...]:

    header = _read_safetensors_header(path)
    ordered = sorted(
        header.items(),
        key=lambda item: (
            item[1]["data_offsets"][0],
            item[1]["data_offsets"][1],
            item[0],
        ),
    )
    previous_end = 0
    for name, entry in ordered:
        start, end = entry["data_offsets"]
        if start < previous_end:
            raise PackedCheckpointError(
                f"overlapping data_offsets before {name} in {path}"
            )
        previous_end = end
    return tuple(name for name, _ in ordered)


def _read_source_tensor(handle: Any, name: str) -> torch.Tensor:

    return handle.get_tensor(name)


@dataclass(frozen=True, slots=True)
class _PackedFileRecord:
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedQwen3Checkpoint:

    root: Path
    manifest: dict[str, Any]
    census: ModelCensus
    packed_expert_elements: int
    _layer_files: tuple[_PackedFileRecord, ...]
    _nonexpert_file: _PackedFileRecord

    @property
    def nonexpert_path(self) -> Path:
        return self.root / self._nonexpert_file.filename

    def layer_path(self, layer_id: int) -> Path:
        if not 0 <= layer_id < len(self._layer_files):
            raise IndexError(
                f"layer_id must be in [0, {len(self._layer_files) - 1}]"
            )
        return self.root / self._layer_files[layer_id].filename

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        source_layout: CheckpointLayout | None = None,
        source_metadata_root: str | Path | None = None,
        verify_hashes: bool = False,
    ) -> PackedQwen3Checkpoint:
        if source_layout is not None and source_metadata_root is not None:
            raise ValueError(
                "source_layout and source_metadata_root are mutually exclusive"
            )
        checkpoint_root = Path(root).resolve()
        manifest_path = checkpoint_root / PACKED_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise PackedCheckpointError(
                f"packed checkpoint manifest is missing: {manifest_path}"
            )
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise PackedCheckpointError(
                f"packed checkpoint manifest is unreadable: {manifest_path}"
            ) from error
        if not isinstance(manifest, dict):
            raise PackedCheckpointError(
                "packed checkpoint manifest must be an object"
            )
        if manifest.get("complete") is not True:
            raise PackedCheckpointError(
                "packed checkpoint manifest must declare complete=true"
            )
        if manifest.get("schema") != PACKED_CHECKPOINT_SCHEMA:
            raise PackedCheckpointError(
                "packed checkpoint manifest schema does not match"
            )
        if manifest.get("schema_version") != PACKED_CHECKPOINT_VERSION:
            raise PackedCheckpointError(
                "packed checkpoint manifest schema version does not match"
            )

        source = _require_mapping(manifest, "source")
        if _require_string(source, "config_filename") != "config.json":
            raise PackedCheckpointError(
                "source config filename does not match"
            )
        if (
            _require_string(source, "index_filename")
            != "model.safetensors.index.json"
        ):
            raise PackedCheckpointError(
                "source index filename does not match"
            )
        source_config_sha256 = _require_sha256(
            source,
            "config_sha256",
        )
        source_index_sha256 = _require_sha256(
            source,
            "index_sha256",
        )
        for provenance_key in ("repo_id", "revision"):
            if provenance_key in source:
                _require_string(source, provenance_key)
        model = _require_mapping(manifest, "model")
        census_mapping = _require_mapping(model, "census")
        census = _parse_census(census_mapping)
        packed_expert_elements = _require_positive_int(
            model,
            "packed_expert_elements",
        )
        expert_dtype = _require_string(model, "expert_dtype")
        try:
            dtype_bytes = _TORCH_DTYPES[expert_dtype].itemsize
        except KeyError as error:
            raise PackedCheckpointError(
                f"unsupported packed expert dtype: {expert_dtype}"
            ) from error
        if model.get("expert_layout") != list(PACKED_EXPERT_LAYOUT):
            raise PackedCheckpointError(
                "packed checkpoint expert layout does not match"
            )
        expected_elements = (
            3 * census.hidden_size * census.moe_intermediate_size
        )
        if packed_expert_elements != expected_elements:
            raise PackedCheckpointError(
                "packed expert element count does not match model dimensions"
            )
        if packed_expert_elements * dtype_bytes != census.expert_bytes:
            raise PackedCheckpointError(
                "packed expert byte count does not match census"
            )
        if census.expert_count != (
            census.num_hidden_layers * census.num_experts
        ):
            raise PackedCheckpointError(
                "packed expert count does not match census dimensions"
            )
        if census.routed_expert_bytes != (
            census.expert_count * census.expert_bytes
        ):
            raise PackedCheckpointError(
                "routed expert byte count does not match census"
            )
        if census.checkpoint_total_bytes != (
            census.routed_expert_bytes + census.nonexpert_bytes
        ):
            raise PackedCheckpointError(
                "checkpoint byte count does not match census"
            )

        files = _require_mapping(manifest, "files")
        nonexpert = _parse_file_record(
            _require_mapping(files, "nonexperts"),
            label="nonexperts",
        )
        raw_layers = files.get("layers")
        if not isinstance(raw_layers, list):
            raise PackedCheckpointError(
                "packed checkpoint layer files must be a list"
            )
        if len(raw_layers) != census.num_hidden_layers:
            raise PackedCheckpointError(
                "packed checkpoint layer file count does not match census"
            )
        layer_files: list[_PackedFileRecord] = []
        for expected_layer_id, raw_record in enumerate(raw_layers):
            if not isinstance(raw_record, dict):
                raise PackedCheckpointError(
                    "packed checkpoint layer file record must be an object"
                )
            if raw_record.get("layer_id") != expected_layer_id:
                raise PackedCheckpointError(
                    "packed checkpoint layer IDs must be contiguous"
                )
            if raw_record.get("tensor_name") != "experts":
                raise PackedCheckpointError(
                    "packed checkpoint layer tensor name must be experts"
                )
            if raw_record.get("shape") != [
                census.num_experts,
                packed_expert_elements,
            ]:
                raise PackedCheckpointError(
                    "packed checkpoint layer tensor shape does not match"
                )
            if raw_record.get("dtype") != expert_dtype:
                raise PackedCheckpointError(
                    "packed checkpoint layer tensor dtype does not match"
                )
            layer_files.append(
                _parse_file_record(
                    raw_record,
                    label=f"layer {expected_layer_id}",
                )
            )

        records = (nonexpert, *layer_files)
        filenames = [record.filename for record in records]
        if len(filenames) != len(set(filenames)):
            raise PackedCheckpointError(
                "packed checkpoint file names must be unique"
            )
        for record in records:
            path = checkpoint_root / record.filename
            if not path.is_file():
                raise PackedCheckpointError(
                    f"packed checkpoint file is missing: {path}"
                )
            if path.stat().st_size != record.size_bytes:
                raise PackedCheckpointError(
                    f"packed checkpoint file size does not match: {path}"
                )
            if verify_hashes and _sha256(path) != record.sha256:
                raise PackedCheckpointError(
                    f"packed checkpoint file SHA256 does not match: {path}"
                )

        if source_layout is not None:
            if (
                source_config_sha256 != source_layout.config_sha256
            ):
                raise PackedCheckpointError(
                    "source config SHA256 does not match packed checkpoint"
                )
            if source_index_sha256 != source_layout.index_sha256:
                raise PackedCheckpointError(
                    "source index SHA256 does not match packed checkpoint"
                )
            if census.as_dict() != source_layout.census.as_dict():
                raise PackedCheckpointError(
                    "source census does not match packed checkpoint"
                )
        elif source_metadata_root is not None:
            metadata_root = Path(source_metadata_root).resolve()
            config_path = metadata_root / "config.json"
            index_path = metadata_root / "model.safetensors.index.json"
            if not config_path.is_file() or not index_path.is_file():
                raise PackedCheckpointError(
                    "source config or index metadata is missing"
                )
            if _sha256(config_path) != source_config_sha256:
                raise PackedCheckpointError(
                    "source config SHA256 does not match packed checkpoint"
                )
            if _sha256(index_path) != source_index_sha256:
                raise PackedCheckpointError(
                    "source index SHA256 does not match packed checkpoint"
                )

        return cls(
            root=checkpoint_root,
            manifest=manifest,
            census=census,
            packed_expert_elements=packed_expert_elements,
            _layer_files=tuple(layer_files),
            _nonexpert_file=nonexpert,
        )


def _require_mapping(
    container: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise PackedCheckpointError(
            f"packed checkpoint manifest field {key!r} must be an object"
        )
    return value


def _require_positive_int(container: dict[str, Any], key: str) -> int:
    value = container.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PackedCheckpointError(
            f"packed checkpoint manifest field {key!r} "
            "must be a positive integer"
        )
    return value


def _require_nonnegative_int(container: dict[str, Any], key: str) -> int:
    value = container.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PackedCheckpointError(
            f"packed checkpoint manifest field {key!r} "
            "must be a nonnegative integer"
        )
    return value


def _require_string(container: dict[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise PackedCheckpointError(
            f"packed checkpoint manifest field {key!r} "
            "must be a nonempty string"
        )
    return value


def _require_sha256(container: dict[str, Any], key: str) -> str:
    digest = _require_string(container, key)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise PackedCheckpointError(
            f"packed checkpoint manifest field {key!r} "
            "must be a lowercase SHA256"
        )
    return digest


def _parse_census(raw: dict[str, Any]) -> ModelCensus:
    return ModelCensus(
        model_type=_require_string(raw, "model_type"),
        torch_dtype=_require_string(raw, "torch_dtype"),
        num_hidden_layers=_require_positive_int(raw, "num_hidden_layers"),
        num_experts=_require_positive_int(raw, "num_experts"),
        num_experts_per_tok=_require_positive_int(
            raw,
            "num_experts_per_tok",
        ),
        hidden_size=_require_positive_int(raw, "hidden_size"),
        moe_intermediate_size=_require_positive_int(
            raw,
            "moe_intermediate_size",
        ),
        shared_expert_tensor_count=_require_nonnegative_int(
            raw,
            "shared_expert_tensor_count",
        ),
        checkpoint_total_bytes=_require_positive_int(
            raw,
            "checkpoint_total_bytes",
        ),
        routed_expert_bytes=_require_positive_int(
            raw,
            "routed_expert_bytes",
        ),
        nonexpert_bytes=_require_nonnegative_int(raw, "nonexpert_bytes"),
        expert_bytes=_require_positive_int(raw, "expert_bytes"),
        expert_count=_require_positive_int(raw, "expert_count"),
    )


def _parse_file_record(
    raw: dict[str, Any],
    *,
    label: str,
) -> _PackedFileRecord:
    filename = _require_string(raw, "filename")
    relative = Path(filename)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative == Path(".")
    ):
        raise PackedCheckpointError(
            f"packed checkpoint {label} file name must stay under root"
        )
    digest = _require_sha256(raw, "sha256")
    return _PackedFileRecord(
        filename=filename,
        size_bytes=_require_positive_int(raw, "size_bytes"),
        sha256=digest,
    )


def _build_expert_destinations(
    layout: CheckpointLayout,
) -> tuple[
    dict[str, tuple[int, int, int, int]],
    int,
    str,
]:
    destinations: dict[str, tuple[int, int, int, int]] = {}
    expert_dtypes = {
        location.dtype
        for expert in layout.experts
        for location in (
            expert.gate_proj,
            expert.up_proj,
            expert.down_proj,
        )
    }
    if len(expert_dtypes) != 1:
        raise PackedCheckpointError(
            f"routed expert dtypes are not uniform: {expert_dtypes}"
        )
    expert_dtype = next(iter(expert_dtypes))
    try:
        torch_dtype = _TORCH_DTYPES[expert_dtype]
    except KeyError as error:
        raise PackedCheckpointError(
            f"unsupported routed expert dtype: {expert_dtype}"
        ) from error

    expected_elements = (
        3
        * layout.census.hidden_size
        * layout.census.moe_intermediate_size
    )
    for expert in layout.experts:
        offset = 0
        for location in (
            expert.gate_proj,
            expert.up_proj,
            expert.down_proj,
        ):
            elements = location.nbytes // torch_dtype.itemsize
            destinations[location.name] = (
                expert.key.layer_id,
                expert.key.expert_id,
                offset,
                elements,
            )
            offset += elements
        if offset != expected_elements:
            raise PackedCheckpointError(
                f"packed size does not match expert {expert.key}"
            )
    return destinations, expected_elements, expert_dtype


def _load_source_tensors(
    source_root: Path,
    layout: CheckpointLayout,
    expert_buffers: list[torch.Tensor],
    destinations: dict[str, tuple[int, int, int, int]],
) -> dict[str, torch.Tensor]:
    index_path = source_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PackedCheckpointError(
            "source checkpoint weight_map must be a nonempty object"
        )
    if not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise PackedCheckpointError(
            "source checkpoint weight_map entries must be strings"
        )

    expected_by_shard: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        expected_by_shard.setdefault(shard, set()).add(name)

    nonexperts: dict[str, torch.Tensor] = {}
    seen: set[str] = set()
    for shard_name in sorted(expected_by_shard):
        shard_path = source_root / shard_name
        storage_order = _tensor_names_in_storage_order(shard_path)
        if set(storage_order) != expected_by_shard[shard_name]:
            missing = sorted(expected_by_shard[shard_name] - set(storage_order))
            unexpected = sorted(
                set(storage_order) - expected_by_shard[shard_name]
            )
            raise PackedCheckpointError(
                f"source shard index mismatch for {shard_name}: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        _LOGGER.info(
            "Read source shard %s by physical offset (%d tensors)",
            shard_name,
            len(storage_order),
        )
        with safe_open(
            shard_path,
            framework="pt",
            device="cpu",
        ) as handle:
            for name in storage_order:
                value = _read_source_tensor(handle, name)
                destination = destinations.get(name)
                if destination is None:
                    nonexperts[name] = value.clone()
                else:
                    layer_id, expert_id, offset, elements = destination
                    if value.numel() != elements:
                        raise PackedCheckpointError(
                            f"source tensor element count changed: {name}"
                        )
                    expert_buffers[layer_id][
                        expert_id,
                        offset : offset + elements,
                    ].copy_(value.reshape(-1))
                seen.add(name)

    if seen != set(weight_map):
        missing = sorted(set(weight_map) - seen)
        raise PackedCheckpointError(
            f"source checkpoint tensors were not all read: {missing[:5]}"
        )
    expected_nonexperts = set(weight_map) - set(destinations)
    if set(nonexperts) != expected_nonexperts:
        raise PackedCheckpointError(
            "source nonexpert tensor collection is incomplete"
        )
    if not nonexperts:
        raise PackedCheckpointError(
            "source checkpoint contains no nonexpert tensors"
        )
    return nonexperts


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "filename": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".json.partial")
    rendered = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def pack_qwen3_checkpoint(
    source_root: str | Path,
    output_root: str | Path,
    *,
    source_repo_id: str | None = None,
    source_revision: str | None = None,
) -> PackedQwen3Checkpoint:

    source = Path(source_root).resolve()
    destination = Path(output_root).resolve()
    for name, value in (
        ("source_repo_id", source_repo_id),
        ("source_revision", source_revision),
    ):
        if value is not None and (
            not isinstance(value, str) or not value
        ):
            raise ValueError(f"{name} must be a nonempty string when provided")
    if destination.exists():
        raise FileExistsError(
            f"packed checkpoint destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.partial-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    _LOGGER.info(
        "Derived checkpoint staging directory: %s; kept on build failure, requires manual confirmation before cleanup",
        staging,
    )

    layout = CheckpointLayout.from_pretrained(source)
    destinations, packed_elements, expert_dtype = (
        _build_expert_destinations(layout)
    )
    _LOGGER.info(
        "Allocated %d layers of contiguous expert buffers, %.2f GiB total",
        layout.census.num_hidden_layers,
        layout.census.routed_expert_bytes / 2**30,
    )
    torch_dtype = _TORCH_DTYPES[expert_dtype]
    expert_buffers = [
        torch.empty(
            (layout.census.num_experts, packed_elements),
            dtype=torch_dtype,
            device="cpu",
        )
        for _ in range(layout.census.num_hidden_layers)
    ]
    nonexperts = _load_source_tensors(
        source,
        layout,
        expert_buffers,
        destinations,
    )

    expert_directory = staging / "experts"
    expert_directory.mkdir()
    layer_records = []
    width = max(3, len(str(layout.census.num_hidden_layers - 1)))
    for layer_id, experts in enumerate(expert_buffers):
        layer_path = (
            expert_directory
            / f"layer-{layer_id:0{width}d}.safetensors"
        )
        save_file(
            {"experts": experts},
            layer_path,
            metadata={
                "format": "pt",
                "layer_id": str(layer_id),
                "expert_layout": "|".join(PACKED_EXPERT_LAYOUT),
            },
        )
        record = _file_record(layer_path, staging)
        record.update(
            {
                "layer_id": layer_id,
                "tensor_name": "experts",
                "shape": [
                    layout.census.num_experts,
                    packed_elements,
                ],
                "dtype": expert_dtype,
            }
        )
        layer_records.append(record)
        _LOGGER.info(
            "Wrote expert layer %d/%d: %s",
            layer_id + 1,
            layout.census.num_hidden_layers,
            layer_path.name,
        )

    nonexpert_path = staging / "nonexperts.safetensors"
    save_file(
        nonexperts,
        nonexpert_path,
        metadata={
            "format": "pt",
            "source_config_sha256": layout.config_sha256,
            "source_index_sha256": layout.index_sha256,
        },
    )
    nonexpert_record = _file_record(nonexpert_path, staging)
    nonexpert_record["tensor_count"] = len(nonexperts)
    _LOGGER.info(
        "Wrote non-expert parameters: %s (%d tensors)",
        nonexpert_path.name,
        len(nonexperts),
    )

    source_manifest = {
        "config_filename": "config.json",
        "config_sha256": layout.config_sha256,
        "index_filename": "model.safetensors.index.json",
        "index_sha256": layout.index_sha256,
    }
    if source_repo_id is not None:
        source_manifest["repo_id"] = source_repo_id
    if source_revision is not None:
        source_manifest["revision"] = source_revision

    manifest = {
        "schema": PACKED_CHECKPOINT_SCHEMA,
        "schema_version": PACKED_CHECKPOINT_VERSION,
        "complete": True,
        "source": source_manifest,
        "model": {
            "census": layout.census.as_dict(),
            "packed_expert_elements": packed_elements,
            "expert_layout": list(PACKED_EXPERT_LAYOUT),
            "expert_dtype": expert_dtype,
        },
        "files": {
            "nonexperts": nonexpert_record,
            "layers": layer_records,
        },
    }
    _write_manifest(staging / PACKED_MANIFEST_FILENAME, manifest)
    os.rename(staging, destination)
    _LOGGER.info("Derived checkpoint build complete: %s", destination)
    return PackedQwen3Checkpoint.open(
        destination,
        source_layout=layout,
    )


__all__ = [
    "PACKED_CHECKPOINT_SCHEMA",
    "PACKED_CHECKPOINT_VERSION",
    "PACKED_EXPERT_LAYOUT",
    "PACKED_MANIFEST_FILENAME",
    "PackedCheckpointError",
    "PackedQwen3Checkpoint",
    "pack_qwen3_checkpoint",
]
