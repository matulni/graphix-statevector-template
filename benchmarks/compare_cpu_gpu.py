"""Compare the cuQuantum (GPU) backend against the reference NumPy (CPU) backend.

This is a standalone script (not a ``pytest-benchmark`` suite): GPU timings require explicit
CPU-GPU synchronization, which ``pytest-benchmark`` does not perform. Each pattern is simulated
end-to-end with :meth:`graphix.pattern.Pattern.simulate_pattern`; GPU runs are bracketed by
``cupy.cuda.Device().synchronize()`` so that the measured wall-clock time includes all device work.

Run with::

    uv run python benchmarks/compare_cpu_gpu.py

It writes ``benchmarks/cpu_vs_gpu.png`` and prints a Markdown timing table.
"""

from __future__ import annotations

import timeit
from pathlib import Path
from typing import TYPE_CHECKING

import cupy as cp
import matplotlib.pyplot as plt
from graphix.sim.statevec import StatevectorBackend as CpuBackend
from graphix_mqtbench import Benchmark, BenchmarkName, OptimizationPass
from numpy.random import PCG64, Generator

from graphix_statevec_cuquantum import StatevectorBackend as GpuBackend

if TYPE_CHECKING:
    from collections.abc import Callable

    from graphix.pattern import Pattern

# CPU becomes impractical well before the GPU runs out of memory (4 GB on a GTX 1650).
_CPU_MAX_QUBITS = 14
_GPU_MAX_QUBITS = 20
_QUBITS = tuple(range(2, _GPU_MAX_QUBITS + 1, 2))
_REPEATS = 3


def _best_time(run: Callable[[], object], *, synchronize: bool) -> float:
    """Return the fastest of ``_REPEATS`` wall-clock timings of ``run`` (in seconds)."""
    if synchronize:
        cp.cuda.Device().synchronize()
    return min(timeit.repeat(run, number=1, repeat=_REPEATS))


def _cpu_run(pattern: Pattern) -> Callable[[], object]:
    def run() -> object:
        return pattern.simulate_pattern(backend=CpuBackend(), rng=Generator(PCG64(0)))

    return run


def _gpu_run(pattern: Pattern) -> Callable[[], object]:
    max_space = pattern.max_space()

    def run() -> object:
        result = pattern.simulate_pattern(backend=GpuBackend.with_capacity(max_space), rng=Generator(PCG64(0)))
        cp.cuda.Device().synchronize()
        return result

    return run


def main() -> None:
    """Run the benchmark, print a Markdown table and save the comparison plot."""
    cpu_times: dict[int, float] = {}
    gpu_times: dict[int, float] = {}

    print("| qubits | max space | CPU (ms) | GPU (ms) | speed-up |")
    print("|---|---|---|---|---|")
    for n in _QUBITS:
        pattern = Benchmark(BenchmarkName.QFT, n).to_pattern(OptimizationPass.M)
        gpu_times[n] = _best_time(_gpu_run(pattern), synchronize=True)
        cpu_cell = "--"
        speedup_cell = "--"
        if n <= _CPU_MAX_QUBITS:
            cpu_times[n] = _best_time(_cpu_run(pattern), synchronize=False)
            cpu_cell = f"{cpu_times[n] * 1e3:.1f}"
            speedup_cell = f"{cpu_times[n] / gpu_times[n]:.1f}x"
        print(f"| {n} | {pattern.max_space()} | {cpu_cell} | {gpu_times[n] * 1e3:.1f} | {speedup_cell} |")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(cpu_times), [cpu_times[n] * 1e3 for n in cpu_times], "o--", label="CPU (NumPy)")
    ax.plot(list(gpu_times), [gpu_times[n] * 1e3 for n in gpu_times], "s-", label="GPU (cuQuantum)")
    ax.set_yscale("log")
    ax.set_xlabel("number of qubits (QFT)")
    ax.set_ylabel("simulation time (ms)")
    ax.set_title("MBQC pattern simulation: CPU vs GPU (GTX 1650)")
    ax.legend()
    ax.grid(visible=True, which="both", linewidth=0.3)
    fig.tight_layout()
    output = Path(__file__).with_name("cpu_vs_gpu.png")
    fig.savefig(output, dpi=130)
    print(f"\nPlot saved to {output}")


if __name__ == "__main__":
    main()
