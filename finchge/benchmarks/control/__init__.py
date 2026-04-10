from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.benchmarks.control.cartpole.benchmark import CartPoleBenchmark
    from finchge.benchmarks.control.cartpole.environment import CartPoleEnvironment
    from finchge.benchmarks.control.cartpole.interpreter import CartPoleInterpreter
    from finchge.benchmarks.control.maze.benchmark import MazeBenchmark
    from finchge.benchmarks.control.maze.environment import MazeEnvironment
    from finchge.benchmarks.control.maze.interpreter import MazeInterpreter
    from finchge.benchmarks.control.santafe.benchmark import SantaFeTrailBenchmark
    from finchge.benchmarks.control.santafe.environment import SantaFeEnvironment
    from finchge.benchmarks.control.santafe.interpreter import SantaFeInterpreter


def __getattr__(name: str) -> Any:
    if name in {"SantaFeTrailBenchmark"}:
        from finchge.benchmarks.control.santafe.benchmark import SantaFeTrailBenchmark

        return locals()[name]
    if name in {"MazeBenchmark"}:
        from finchge.benchmarks.control.maze.benchmark import MazeBenchmark

        return locals()[name]
    if name in {"CartPoleBenchmark"}:
        from finchge.benchmarks.control.cartpole.benchmark import CartPoleBenchmark

        return locals()[name]

    if name in {"SantaFeEnvironment"}:
        from finchge.benchmarks.control.santafe.environment import SantaFeEnvironment

        return locals()[name]
    if name in {"MazeEnvironment"}:
        from finchge.benchmarks.control.maze.environment import MazeEnvironment

        return locals()[name]
    if name in {"CartPoleEnvironment"}:
        from finchge.benchmarks.control.cartpole.environment import CartPoleEnvironment

        return locals()[name]

    if name in {"SantaFeInterpreter"}:
        from finchge.benchmarks.control.santafe.interpreter import SantaFeInterpreter

        return locals()[name]
    if name in {"MazeInterpreter"}:
        from finchge.benchmarks.control.maze.interpreter import MazeInterpreter

        return locals()[name]
    if name in {"CartPoleInterpreter"}:
        from finchge.benchmarks.control.cartpole.interpreter import CartPoleInterpreter

        return locals()[name]
    raise AttributeError(name)


__all__ = [
    "SantaFeTrailBenchmark",
    "MazeBenchmark",
    "CartPoleBenchmark",
    "SantaFeEnvironment",
    "MazeEnvironment",
    "CartPoleEnvironment",
    "SantaFeInterpreter",
    "MazeInterpreter",
    "CartPoleInterpreter",
]
