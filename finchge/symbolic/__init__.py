from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.symbolic.expression import SymbolicExpression
    from finchge.symbolic.ge_regressor import GERegressor


def __getattr__(name: str) -> Any:
    if name in {"SymbolicExpression"}:
        from finchge.symbolic.expression import SymbolicExpression

        return locals()[name]
    if name in {"GERegressor"}:
        from finchge.symbolic.ge_regressor import GERegressor

        return locals()[name]
    raise AttributeError(name)


__all__ = ["SymbolicExpression", "GERegressor"]
