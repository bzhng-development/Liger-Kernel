import argparse
from dataclasses import dataclass
from typing import List

import torch
from triton.testing import do_bench

from liger_kernel.ops.geglu_sparse import gelu_and_mul_sparse as ref_gelu_and_mul_sparse
from liger_kernel.ops.geglu_sparse_triton import geglu_sparse_forward as triton_geglu_sparse_forward
from benchmark.utils import (
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    device_name,
    get_quantiles,
    write_benchmark_csv,
    _test_memory,
)


def _make_inputs(B: int, T: int, H: int, dtype: torch.dtype, device: str):
    gate = torch.randn(B, T, H, device=device, dtype=dtype)
    up = torch.randn(B, T, H, device=device, dtype=dtype)
    return gate, up


def _provider_run(provider: str, gate: torch.Tensor, up: torch.Tensor, sparsity: float):
    if provider == "liger_geglu_sparse":
        return triton_geglu_sparse_forward(gate, up, sparsity, approximate="tanh")
    elif provider in ("huggingface", "naive"):
        x = torch.cat([gate, up], dim=-1)
        return ref_gelu_and_mul_sparse(x, activation_sparsity=sparsity, approximate="tanh")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def bench_speed_geglu_sparse(inp: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    B = inp.extra_benchmark_config.get("B", 4)
    T = inp.extra_benchmark_config.get("T", 128)
    H = inp.x_value  # vary hidden size
    dtype = getattr(torch, inp.extra_benchmark_config.get("dtype", "bfloat16"))
    sparsity = float(inp.extra_benchmark_config.get("sparsity", 0.95))

    gate, up = _make_inputs(B, T, H, dtype, inp.device)
    fn = lambda: _provider_run(inp.kernel_provider, gate, up, sparsity)
    # Warmup
    fn(); torch.cuda.synchronize()
    # Timed
    ms = do_bench(fn, rep=200, fast_flush=True)
    return SingleBenchmarkRunOutput(
        kernel_name="geglu_sparse",
        kernel_provider=inp.kernel_provider,
        mode="forward",
        metric="speed",
        value=ms,
        x_axis_name="H",
        x_axis_value=H,
        extra_metadata={
            "B": B,
            "T": T,
            "dtype": str(dtype).replace("torch.", ""),
            "sparsity": sparsity,
        },
        device_name=device_name(),
    )


def bench_memory_geglu_sparse(inp: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    B = inp.extra_benchmark_config.get("B", 4)
    T = inp.extra_benchmark_config.get("T", 128)
    H = inp.x_value
    dtype = getattr(torch, inp.extra_benchmark_config.get("dtype", "bfloat16"))
    sparsity = float(inp.extra_benchmark_config.get("sparsity", 0.95))

    gate, up = _make_inputs(B, T, H, dtype, inp.device)
    def _runner():
        out = _provider_run(inp.kernel_provider, gate, up, sparsity)
        # keep tensor alive
        torch.cuda.synchronize()
        return out

    mem_mb = _test_memory(_runner)
    return SingleBenchmarkRunOutput(
        kernel_name="geglu_sparse",
        kernel_provider=inp.kernel_provider,
        mode="full",
        metric="memory",
        value=mem_mb,
        x_axis_name="H",
        x_axis_value=H,
        extra_metadata={
            "B": B,
            "T": T,
            "dtype": str(dtype).replace("torch.", ""),
            "sparsity": sparsity,
        },
        device_name=device_name(),
    )


def run_benchmarks(output_csv: str, speed: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    providers = ["liger_geglu_sparse", "huggingface"]
    x_values = [256, 512, 1024, 2048, 4096]
    extra_cfg = {"B": 4, "T": 256, "dtype": "bfloat16", "sparsity": 0.95}

    outputs: List[SingleBenchmarkRunOutput] = []
    for provider in providers:
        for x in x_values:
            inp = SingleBenchmarkRunInput(
                kernel_provider=provider,
                device=device,
                x_value=x,
                extra_benchmark_config=extra_cfg,
            )
            out = bench_speed_geglu_sparse(inp) if speed else bench_memory_geglu_sparse(inp)
            outputs.append(out)

    # Write CSV
    write_benchmark_csv(output_csv, outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="benchmark/data/all_benchmark_data.csv")
    args = parser.parse_args()

    # Speed
    run_benchmarks(args.out, speed=True)
    # Memory
    run_benchmarks(args.out, speed=False)

