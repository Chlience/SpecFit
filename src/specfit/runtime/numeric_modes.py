
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


NUMERIC_MODES = (
    "default",
    "math_sdpa",
    "bf16_full_reduction",
    "deterministic",
    "math_ieee",
    "math_bf16_full_reduction",
)

NumericSettings = dict[str, object]


def current_numeric_settings(
    *,
    forced_sdpa_backend: str | None = None,
) -> NumericSettings:

    return {
        "forced_sdpa_backend": forced_sdpa_backend,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp_enabled": (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "cuda_matmul_fp32_precision": (
            torch.backends.cuda.matmul.fp32_precision
        ),
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def _restore_scalar_settings(settings: NumericSettings) -> None:

    torch.backends.cuda.matmul.fp32_precision = str(
        settings["cuda_matmul_fp32_precision"]
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(
        settings["allow_bf16_reduced_precision_reduction"]
    )
    torch.use_deterministic_algorithms(
        bool(settings["deterministic_algorithms"]),
        warn_only=bool(settings["deterministic_warn_only"]),
    )


class NumericModeHandle:

    def __init__(
        self,
        *,
        name: str,
        settings_before: NumericSettings,
        actual_settings: NumericSettings,
        exit_stack: ExitStack,
    ) -> None:
        self.name = name
        self.settings_before = dict(settings_before)
        self.actual_settings = dict(actual_settings)
        self.settings_after: NumericSettings | None = None
        self._exit_stack = exit_stack

    @property
    def restored(self) -> bool:

        return self.settings_after is not None

    @property
    def state_restored(self) -> bool:

        return (
            self.settings_after is not None
            and self.settings_after == self.settings_before
        )

    def restore(self) -> NumericSettings:

        if self.settings_after is None:
            self._exit_stack.close()
            self.settings_after = current_numeric_settings(
                forced_sdpa_backend=None,
            )
        return dict(self.settings_after)

    def artifact(self, *, scope: str) -> dict[str, object]:

        if self.settings_after is None:
            raise RuntimeError(
                "numeric mode must be restored before recording artifact"
            )
        return {
            "name": self.name,
            "scope": scope,
            "settings_before": dict(self.settings_before),
            "actual_settings": dict(self.actual_settings),
            "settings_after": dict(self.settings_after),
            "state_restored": self.state_restored,
        }


def apply_numeric_mode(mode: str) -> NumericModeHandle:

    if mode not in NUMERIC_MODES:
        raise ValueError(f"unsupported numeric mode: {mode}")

    settings_before = current_numeric_settings(forced_sdpa_backend=None)
    stack = ExitStack()
    stack.callback(_restore_scalar_settings, settings_before)
    forced_sdpa_backend: str | None = None

    try:
        if mode in {
            "math_sdpa",
            "math_ieee",
            "math_bf16_full_reduction",
        }:
            stack.enter_context(sdpa_kernel(SDPBackend.MATH))
            forced_sdpa_backend = "math"
        if mode == "math_ieee":
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        if mode in {
            "bf16_full_reduction",
            "math_bf16_full_reduction",
        }:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                False
            )
        if mode == "deterministic":
            torch.use_deterministic_algorithms(True, warn_only=False)

        actual_settings = current_numeric_settings(
            forced_sdpa_backend=forced_sdpa_backend,
        )
    except BaseException:
        stack.close()
        raise

    return NumericModeHandle(
        name=mode,
        settings_before=settings_before,
        actual_settings=actual_settings,
        exit_stack=stack,
    )


@contextmanager
def numeric_mode(mode: str) -> Iterator[NumericSettings]:

    handle = apply_numeric_mode(mode)
    try:
        yield dict(handle.actual_settings)
    finally:
        handle.restore()


__all__ = [
    "NUMERIC_MODES",
    "NumericModeHandle",
    "NumericSettings",
    "apply_numeric_mode",
    "current_numeric_settings",
    "numeric_mode",
]
