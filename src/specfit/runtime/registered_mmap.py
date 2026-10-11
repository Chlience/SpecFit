
from __future__ import annotations

import time
from pathlib import Path

import torch
from safetensors import safe_open

_CUDA_HOST_REGISTER_READ_ONLY = 0x08


class RegisteredMmapExpertLayer:

    __slots__ = (
        "_backing",
        "_closed",
        "_device",
        "_pointer",
        "mapped_bytes",
        "mmap_seconds",
        "registration_seconds",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        expected_shape: tuple[int, int],
        device: torch.device,
    ) -> None:
        mmap_start = time.perf_counter()
        with safe_open(path, framework="pt", device="cpu") as handle:
            if tuple(handle.keys()) != ("experts",):
                raise ValueError(
                    f"packed expert layer must contain only 'experts': {path}"
                )
            backing = handle.get_tensor("experts")
        self.mmap_seconds = time.perf_counter() - mmap_start

        if tuple(backing.shape) != expected_shape:
            raise ValueError(
                f"packed expert layer has shape {tuple(backing.shape)}, "
                f"expected {expected_shape}"
            )
        if backing.dtype is not torch.bfloat16:
            raise ValueError("packed expert layer must use BF16")
        if not backing.is_contiguous():
            raise ValueError("packed expert layer must be contiguous")

        self._backing: torch.Tensor | None = backing
        self._closed = False
        self._device = device
        self._pointer = backing.data_ptr()
        self.mapped_bytes = backing.numel() * backing.element_size()

        registration_start = time.perf_counter()
        with torch.cuda.device(device):
            result = torch.cuda.cudart().cudaHostRegister(
                self._pointer,
                self.mapped_bytes,
                _CUDA_HOST_REGISTER_READ_ONLY,
            )
            torch.cuda.check_error(result)
        self.registration_seconds = (
            time.perf_counter() - registration_start
        )

        if not backing.is_pinned():
            with torch.cuda.device(device):
                result = torch.cuda.cudart().cudaHostUnregister(
                    self._pointer
                )
                torch.cuda.check_error(result)
            self._backing = None
            self._closed = True
            raise RuntimeError(
                "CUDA-registered mmap is not recognized as pinned memory"
            )

    @property
    def tensor(self) -> torch.Tensor:
        if self._closed or self._backing is None:
            raise RuntimeError("registered mmap expert layer is closed")
        return self._backing

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:

        if self._closed:
            return
        with torch.cuda.device(self._device):
            result = torch.cuda.cudart().cudaHostUnregister(self._pointer)
            torch.cuda.check_error(result)
        self._backing = None
        self._closed = True
