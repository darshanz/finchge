from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseAlgorithm, BaseAlgorithmMO
    from .ga import GeneticAlgorithm
    from .nsga import NSGA2, NSGA3


def __getattr__(name: str) -> Any:
    if name in {"BaseAlgorithm", "BaseAlgorithmMO"}:
        from .base import BaseAlgorithm, BaseAlgorithmMO

    if name in {"GeneticAlgorithm"}:
        from .ga import GeneticAlgorithm

        return locals()[name]
    if name in {"NSGA2", "NSGA3"}:
        from .nsga import NSGA2, NSGA3

        return locals()[name]
    raise AttributeError(name)


__all__ = ["BaseAlgorithm", "BaseAlgorithmMO", "GeneticAlgorithm", "NSGA2", "NSGA3"]
