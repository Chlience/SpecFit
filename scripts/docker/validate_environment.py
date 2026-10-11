#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


LOCK_PATH = Path("/opt/specfit/base-packages.lock.json")


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "MISSING"


def validate_cpu() -> dict[str, object]:
    expected = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    actual_packages = {
        name: package_version(name) for name in expected["packages"] if name != "triton"
    }
    triton = importlib.import_module("triton")
    actual_packages["triton"] = triton.__version__

    mismatches = {
        name: {"expected": expected_version, "actual": actual_packages.get(name)}
        for name, expected_version in expected["packages"].items()
        if actual_packages.get(name) != expected_version
    }
    if mismatches:
        raise RuntimeError(f"Base package mismatch: {json.dumps(mismatches, indent=2)}")

    if platform.python_version() != expected["python"]:
        raise RuntimeError(
            f"Python mismatch: expected {expected['python']}, got {platform.python_version()}"
        )
    if os.environ.get("CUDA_VERSION") != expected["cuda"]:
        raise RuntimeError(
            f"CUDA mismatch: expected {expected['cuda']}, got {os.environ.get('CUDA_VERSION')}"
        )

    import accelerate
    import torch
    import transformer_engine.pytorch as te
    import transformers
    from speculators.models.eagle3 import Eagle3DraftModel
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    known_base_issue = (
        "nvidia-resiliency-ext 0.4.1+cuda13 requires pynvml, which is not installed."
    )
    pip_errors = [
        line
        for line in (pip_check.stdout + pip_check.stderr).splitlines()
        if line and line != known_base_issue
    ]
    if pip_errors:
        raise RuntimeError("\n".join(pip_errors))

    return {
        "python": platform.python_version(),
        "cuda": os.environ.get("CUDA_VERSION"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "transformer_engine_module": te.__name__,
        "qwen_config": Qwen3MoeConfig.__name__,
        "qwen_model": Qwen3MoeForCausalLM.__name__,
        "eagle_model": Eagle3DraftModel.__name__,
        "protected_packages": actual_packages,
    }


def validate_gpu(compile_extension: bool) -> dict[str, object]:
    import torch
    import transformer_engine.pytorch as te

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")

    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 0):
        raise RuntimeError(f"Expected A100 compute capability 8.0, got {capability}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable")

    lhs = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
    rhs = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
    product = lhs @ rhs
    if not torch.isfinite(product).all():
        raise RuntimeError("PyTorch BF16 GEMM produced non-finite values")

    te_input = torch.randn(5, 16, device=device, dtype=torch.bfloat16)
    te_linear = te.Linear(16, 8, params_dtype=torch.bfloat16).cuda()
    te_grouped = te.GroupedLinear(2, 16, 8, params_dtype=torch.bfloat16).cuda()
    linear_output = te_linear(te_input)
    grouped_output = te_grouped(te_input, [2, 3])
    if not torch.isfinite(linear_output).all() or not torch.isfinite(grouped_output).all():
        raise RuntimeError("Transformer Engine produced non-finite values")

    host = torch.empty(9 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    copied = host.to(device, non_blocking=True)
    end.record()
    end.synchronize()

    result: dict[str, object] = {
        "gpu_name": name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "bf16": True,
        "torch_bf16_gemm": list(product.shape),
        "te_linear": list(linear_output.shape),
        "te_grouped_linear": list(grouped_output.shape),
        "h2d_9mib_ms": start.elapsed_time(end),
        "h2d_bytes": copied.numel(),
    }

    if compile_extension:
        result["cuda_extension"] = compile_cuda_extension()
    return result


def compile_cuda_extension() -> str:
    import torch
    from torch.utils.cpp_extension import load_inline

    cpp_source = """
    #include <torch/extension.h>
    torch::Tensor add_one_cuda(torch::Tensor input);
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("add_one", &add_one_cuda, "add one on CUDA");
    }
    """
    cuda_source = """
    #include <torch/extension.h>
    __global__ void add_one_kernel(float* values, int64_t count) {
      const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index < count) values[index] += 1.0f;
    }
    torch::Tensor add_one_cuda(torch::Tensor input) {
      auto output = input.clone();
      const int threads = 256;
      const int blocks = (output.numel() + threads - 1) / threads;
      add_one_kernel<<<blocks, threads>>>(output.data_ptr<float>(), output.numel());
      return output;
    }
    """
    extension = load_inline(
        name="specfit_cuda_smoke",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=None,
        extra_cuda_cflags=["-O2"],
        with_cuda=True,
        verbose=False,
    )
    values = torch.zeros(1024, device="cuda")
    output = extension.add_one(values)
    torch.cuda.synchronize()
    if not torch.equal(output, torch.ones_like(output)):
        raise RuntimeError("CUDA extension produced an incorrect result")
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Run CPU/import checks")
    parser.add_argument("--gpu", action="store_true", help="Run A100 GPU checks")
    parser.add_argument("--compile-extension", action="store_true")
    args = parser.parse_args()

    result: dict[str, object] = {}
    if args.cpu or not args.gpu:
        result["cpu"] = validate_cpu()
    if args.gpu:
        result["gpu"] = validate_gpu(args.compile_extension)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
