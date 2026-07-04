from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseAlgorithm, BaseAlgorithmMO, BaseAlgorithmSO
    from .ga import GeneticAlgorithm
    from .nsga import NSGA2, NSGA3


def __getattr__(name: str) -> Any:
    if name in {"BaseAlgorithm", "BaseAlgorithmSO", "BaseAlgorithmMO"}:
        from .base import BaseAlgorithm, BaseAlgorithmMO, BaseAlgorithmSO

        return locals()[name]

    if name in {"GeneticAlgorithm"}:
        from .ga import GeneticAlgorithm

        return locals()[name]

    if name in {"NSGA2", "NSGA3"}:
        from .nsga import NSGA2, NSGA3

        return locals()[name]

    raise AttributeError(name)


__all__ = [
    "BaseAlgorithm",
    "BaseAlgorithmSO",
    "BaseAlgorithmMO",
    "GeneticAlgorithm",
    "NSGA2",
    "NSGA3",
]
