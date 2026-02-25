from typing import Any, Dict, List, Optional, Type

from .base import Benchmark


class BenchmarkRegistry:
    """
    Registry for benchmark classes.
    """

    _benchmarks: Dict[str, Type[Benchmark]] = {}

    @classmethod
    def register(cls, name: str, category: str) -> Any:
        """Decorator to register a benchmark."""

        def decorator(benchmark_class: Any) -> Any:
            key = f"{category}.{name}"
            cls._benchmarks[key] = benchmark_class
            return benchmark_class

        return decorator

    @classmethod
    def get(cls, name: str, category: Optional[str] = None) -> Benchmark:
        """
        Get a benchmark instance by name.

        Args:
            name: Benchmark name
            category: Optional category (if not provided, searches all)
            **kwargs: Passed to benchmark constructor

        Returns:
            Benchmark instance
        """
        if category:
            key = f"{category}.{name}"
            if key not in cls._benchmarks:
                raise ValueError(f"Benchmark '{key}' not found")
            return cls._benchmarks[key]()

        # Search all categories
        for key, bench_class in cls._benchmarks.items():
            if key.endswith(f".{name}"):
                return bench_class()

        raise ValueError(f"Benchmark '{name}' not found in any category")

    @classmethod
    def list_benchmarks(cls, category: Optional[str] = None) -> List[str]:
        """List all available benchmarks."""
        if category:
            return [
                k.split(".")[1]
                for k in cls._benchmarks.keys()
                if k.startswith(f"{category}.")
            ]
        return list(cls._benchmarks.keys())


# Convenience functions
register = BenchmarkRegistry.register
get_benchmark = BenchmarkRegistry.get
list_benchmarks = BenchmarkRegistry.list_benchmarks
