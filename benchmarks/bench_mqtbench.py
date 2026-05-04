# This module benchmarks the statevector simulator against the MQTBench suite
# It relies on the plugin graphix-mqtbench which is not stable yet.

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from graphix_mqtbench import Benchmark, BenchmarkName, BenchmarkRunner, OptimizationPass

from graphix_statevec_template import StatevectorBackend

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


@pytest.mark.skip(reason="Not Implemented")
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
        if mqt_benchmark is not None:
            runner = BenchmarkRunner(
                benchmark=mqt_benchmark,
                benchmark_fixture=benchmark,
                optim=OptimizationPass.M,
                backend_generator=lambda _: StatevectorBackend(),
                backend_name="statevector_template",
            )
            runner.run()  # type: ignore[no-untyped-call] # graphix-mqtbench is not annotated.
