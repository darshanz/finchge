from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseAlgorithm, BaseAlgorithmMO, BaseAlgorithmSO
    from .clonalg import CLONALG
    from .ga import GeneticAlgorithm
    from .island_ga import IslandGA
    from .memetic import MemeticGA
    from .mu_comma_lambda import MuCommaLambdaES
    from .mu_plus_lambda import MuPlusLambdaES
    from .nsga import NSGA2, NSGA3
    from .one_plus_one import OnePlusOneES
    from .steady_state_ga import SteadyStateGA


def __getattr__(name: str) -> Any:
    if name in {"BaseAlgorithm", "BaseAlgorithmSO", "BaseAlgorithmMO"}:
        from .base import BaseAlgorithm, BaseAlgorithmMO, BaseAlgorithmSO

        return locals()[name]

    if name in {"GeneticAlgorithm"}:
        from .ga import GeneticAlgorithm

        return locals()[name]

    if name in {"SteadyStateGA"}:
        from .steady_state_ga import SteadyStateGA

        return locals()[name]

    if name in {"IslandGA"}:
        from .island_ga import IslandGA

        return locals()[name]

    if name in {"OnePlusOneES"}:
        from .one_plus_one import OnePlusOneES

        return locals()[name]

    if name in {"MuPlusLambdaES"}:
        from .mu_plus_lambda import MuPlusLambdaES

        return locals()[name]

    if name in {"MuCommaLambdaES"}:
        from .mu_comma_lambda import MuCommaLambdaES

        return locals()[name]

    if name in {"MemeticGA"}:
        from .memetic import MemeticGA

        return locals()[name]

    if name in {"CLONALG"}:
        from .clonalg import CLONALG

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
    "SteadyStateGA",
    "IslandGA",
    "OnePlusOneES",
    "MuPlusLambdaES",
    "MuCommaLambdaES",
    "MemeticGA",
    "CLONALG",
    "NSGA2",
    "NSGA3",
]
