from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.benchmarks.regression.keijzer.benchmark import KeijzerBenchmark
    from finchge.benchmarks.regression.koza_quartic.benchmark import (
        KozaQuarticBenchmark,
    )
    from finchge.benchmarks.regression.nguyen.benchmark import NguyenBenchmark
    from finchge.benchmarks.regression.vladislavleva.benchmark import (
        VladislavlevaBenchmark,
    )


def __getattr__(name: str) -> Any:
    if name in {"KozaQuarticBenchmark"}:
        from finchge.benchmarks.regression.koza_quartic.benchmark import (
            KozaQuarticBenchmark,
        )

        return locals()[name]
    if name in {"NguyenBenchmark"}:
        from finchge.benchmarks.regression.nguyen.benchmark import NguyenBenchmark

        return locals()[name]
    if name in {"KeijzerBenchmark"}:
        from finchge.benchmarks.regression.keijzer.benchmark import KeijzerBenchmark

        return locals()[name]
    if name in {"VladislavlevaBenchmark"}:
        from finchge.benchmarks.regression.vladislavleva.benchmark import (
            VladislavlevaBenchmark,
        )

        return locals()[name]
    raise AttributeError(name)


__all__ = [
    "KozaQuarticBenchmark",
    "NguyenBenchmark",
    "KeijzerBenchmark",
    "VladislavlevaBenchmark",
]
