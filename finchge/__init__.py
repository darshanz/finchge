__version__ = "1.0.1-beta.15"

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.config import FinchConfig, Keys
    from finchge.core import GrammaticalEvolution
    from finchge.fitness import FitnessEvaluator, GEFitnessFunction
    from finchge.grammar import GenotypeMapper, Grammar


def __getattr__(name: str) -> Any:
    if name in {"FinchConfig", "Keys"}:
        from finchge.config import FinchConfig, Keys

        return locals()[name]
    if name == "GrammaticalEvolution":
        from finchge.core import GrammaticalEvolution

        return GrammaticalEvolution
    if name in {"FitnessEvaluator", "GEFitnessFunction"}:
        from finchge.fitness import FitnessEvaluator, GEFitnessFunction

        return locals()[name]
    if name in {"Grammar", "GenotypeMapper"}:
        from finchge.grammar import GenotypeMapper, Grammar

        return locals()[name]
    raise AttributeError(name)


__all__ = [
    "__version__",
    "FinchConfig",
    "Keys",
    "GrammaticalEvolution",
    "FitnessEvaluator",
    "GEFitnessFunction",
    "Grammar",
    "GenotypeMapper",
    "algorithm",
]
