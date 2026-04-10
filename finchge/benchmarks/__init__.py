from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
    from finchge.benchmarks.control import (  # Base classes
        ControlEnvironment,
        SantaFeEnvironment,
        SantaFeTrailBenchmark,
    )


def __getattr__(name: str) -> Any:
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
    "ControlEnvironment",
    # Santa Fe
    "SantaFeEnvironment",
    "SantaFeTrailBenchmark",
]
