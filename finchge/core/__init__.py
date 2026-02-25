from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.core.engine import GrammaticalEvolution
    from finchge.core.individual import Individual
    from finchge.core.population import Population


def __getattr__(name: str) -> Any:
    if name in {"GrammaticalEvolution"}:
        from finchge.core.engine import GrammaticalEvolution

        return locals()[name]
    if name in {"Individual"}:
        from finchge.core.individual import Individual

        return locals()[name]

    if name in {"Population"}:
        from finchge.core.population import Population

        return locals()[name]

    raise AttributeError(name)


__all__ = ["GrammaticalEvolution", "Individual", "Population"]
