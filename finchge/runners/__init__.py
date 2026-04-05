from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.runners.base import PhenotypeRunner
    from finchge.runners.control import ControlRunner
    from finchge.runners.logic import LogicRunner
    from finchge.runners.ml import MLModelRunner
    from finchge.runners.sr import SymbolicRegressionRunner


def __getattr__(name: str) -> Any:
    if name in {"PhenotypeRunner"}:
        from finchge.runners.base import PhenotypeRunner

        return locals()[name]
    if name in {"ControlRunner"}:
        from finchge.runners.control import ControlRunner

        return locals()[name]
    if name in {"LogicRunner"}:
        from finchge.runners.logic import LogicRunner

        return locals()[name]
    if name in {"MLModelRunner"}:
        from finchge.runners.ml import MLModelRunner

        return locals()[name]
    if name in {"SymbolicRegressionRunner"}:
        from finchge.runners.sr import SymbolicRegressionRunner

        return locals()[name]
    raise AttributeError(name)


__all__ = [
    "PhenotypeRunner",
    "ControlRunner",
    "LogicRunner",
    "MLModelRunner",
    "SymbolicRegressionRunner",
]
