from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.benchmarks.logic.interpreters import LogicInterpreter
    from finchge.benchmarks.logic.multiplexer.benchmark import MultiplexerBenchmark


def __getattr__(name: str) -> Any:
    if name in {"MultiplexerBenchmark"}:
        from finchge.benchmarks.logic.multiplexer.benchmark import MultiplexerBenchmark

        return locals()[name]
    if name in {"LogicInterpreter"}:
        from finchge.benchmarks.logic.interpreters import LogicInterpreter

        return locals()[name]
    raise AttributeError(name)


__all__ = [
    "MultiplexerBenchmark",
    "LogicInterpreter",
]
