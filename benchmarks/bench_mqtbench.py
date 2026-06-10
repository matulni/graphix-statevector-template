# This module benchmarks the statevector simulator against the MQTBench suite
# It relies on the plugin graphix-mqtbench which is not stable yet.

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from graphix_mqtbench import Benchmark, BenchmarkName, BenchmarkRunner, OptimizationPass

from graphix_statevec_cuquantum import StatevectorBackend
from graphix_statevec_cuquantum.graphix_statevec_cuquantum import _gpu_available

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture

pytestmark = pytest.mark.skipif(not _gpu_available(), reason="GPU not available")


class BenchTest:
    _BENCHMARKS = (
        Benchmark(BenchmarkName.FULL_ADDER, 16),
        Benchmark(BenchmarkName.QFT, 16),
        Benchmark(BenchmarkName.RANDOMCIRCUIT, 16),
    )
    group = "mqtbench"

    @pytest.mark.benchmark(group=group, max_time=1)
    @pytest.mark.parametrize("mqt_benchmark", _BENCHMARKS)
    def bench_statevector(self, benchmark: BenchmarkFixture, mqt_benchmark: Benchmark) -> None:
        runner = BenchmarkRunner(
            benchmark=mqt_benchmark,
            benchmark_fixture=benchmark,
            optim=OptimizationPass.M,
            # Preallocate the GPU buffer to the pattern's maximum space.
            backend_generator=lambda pattern: StatevectorBackend.with_capacity(pattern.max_space()),
            backend_name="cuquantum",
        )
        runner.run()  # type: ignore[no-untyped-call] # graphix-mqtbench is not annotated.
