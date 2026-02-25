from typing import TYPE_CHECKING, Any


def __getattr__(name: str) -> Any:
    if name in {"get_benchmark", "list_benchmarks", "register"}:
        from .registry import get_benchmark, list_benchmarks, register

        return locals()[name]
    if name in {"Benchmark", "BenchmarkMetadata"}:
        from .base import Benchmark, BenchmarkMetadata

        return locals()[name]
    if name in {"ControlEnvironment", "SantaFeEnvironment", "SantaFeTrailBenchmark"}:
        from finchge.benchmarks.control import (  # Base classes
            ControlEnvironment,
            SantaFeEnvironment,
            SantaFeTrailBenchmark,
        )

        return locals()[name]
    raise AttributeError(name)


__all__ = [
    "Benchmark",
    "BenchmarkMetadata",
    "get_benchmark",
    "list_benchmarks",
    "register",
    # Control base
    "ControlEnvironment"
    # Santa Fe
    "SantaFeEnvironment",
    "SantaFeTrailBenchmark",
]
