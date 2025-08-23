import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.ops.geglu_sparse import gelu_and_mul_sparse as ref_gelu_and_mul_sparse
from liger_kernel.ops.geglu_sparse_triton import (
    geglu_sparse_forward as triton_geglu_sparse_forward,
)
from liger_kernel.utils import infer_device

device = infer_device()


def _make_inputs(B: int, T: int, H: int, dtype: torch.dtype):
    gate = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
    up = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
    return gate, up


def _provider_run(provider: str, gate: torch.Tensor, up: torch.Tensor, sparsity: float):
    if provider == "liger":
        return triton_geglu_sparse_forward(gate, up, sparsity, approximate="tanh")
    elif provider in ("huggingface", "naive"):
        x = torch.cat([gate, up], dim=-1)
        return ref_gelu_and_mul_sparse(x, activation_sparsity=sparsity, approximate="tanh")
    else:
        raise ValueError(f"Invalid provider: {provider} for geglu_sparse")


def bench_speed_geglu_sparse(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    H = int(input.x)
    B = int(input.extra_benchmark_config.get("B", 4))
    T = int(input.extra_benchmark_config.get("T", 128))
    dtype = input.extra_benchmark_config.get("dtype", torch.bfloat16)
    sparsity = float(input.extra_benchmark_config.get("sparsity", 0.95))
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    gate, up = _make_inputs(B, T, H, dtype)

    def fwd():
        return _provider_run(provider, gate, up, sparsity)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            grad_to_none=[gate, up],
            rep=10,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        do = torch.randn(B, T, H, device=device, dtype=dtype)
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(do, retain_graph=True),
            grad_to_none=[gate, up],
            rep=10,
            quantiles=QUANTILES,
        )
    else:

        def full():
            y = fwd()
            y.backward(torch.randn_like(y), retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[gate, up],
            rep=10,
            quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_geglu_sparse(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    H = int(input.x)
    B = int(input.extra_benchmark_config.get("B", 4))
    T = int(input.extra_benchmark_config.get("T", 128))
    dtype = input.extra_benchmark_config.get("dtype", torch.bfloat16)
    sparsity = float(input.extra_benchmark_config.get("sparsity", 0.95))
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    gate, up = _make_inputs(B, T, H, dtype)

    def fwd():
        return _provider_run(provider, gate, up, sparsity)

    def full():
        y = fwd()
        y.backward(torch.randn_like(y), retain_graph=True)

    if mode == "forward":
        mem_50, mem_20, mem_80 = _test_memory(
            fwd,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = fwd()
        do = torch.randn(B, T, H, device=device, dtype=dtype)
        mem_50, mem_20, mem_80 = _test_memory(
            lambda: y.backward(do, retain_graph=True),
            quantiles=QUANTILES,
        )
    else:
        mem_50, mem_20, mem_80 = _test_memory(
            full,
            quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "geglu_sparse",
        "x_name": "H",
        "x_label": "hidden size",
        "x_values": [256, 512, 1024, 2048, 4096],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "B": 4,
                "T": 256,
                "dtype": torch.bfloat16,
                "sparsity": 0.95,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_geglu_sparse,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_geglu_sparse,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
