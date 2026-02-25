from typing import Any, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


def _and_op(a: bool, b: bool) -> bool:
    return a and b


def _or_op(a: bool, b: bool) -> bool:
    return a or b


def _not_op(a: bool) -> bool:
    return not a


def _if_op(c: bool, t: T, f: T) -> T:
    return t if c else f


class LogicInterpreter:
    """
    Interpreter for logic programs with IF, AND, OR, NOT.

    Grammar:
        <expr> ::= <if> | <and> | <or> | <not> | <var>
        <if>  ::= if ( <expr> , <expr> , <expr> )
        <and> ::= and ( <expr> , <expr> )
        <or>  ::= or ( <expr> , <expr> )
        <not> ::= not ( <expr> )
        <var> ::= x0 | x1 | x2 | x3 | ...

    Notes:
        - Variables are positional inputs (x0 → inputs[0]).
        - Uses restricted `eval` with a controlled namespace.
        - Pickle-safe (no lambdas).
    """

    def __init__(self) -> None:
        self.operations = {
            "and": _and_op,
            "or": _or_op,
            "not": _not_op,
            "if": _if_op,
        }

    def tokenize(self, program: str) -> list[str]:
        import re

        tokens = re.findall(r"if|and|or|not|x\d+|[(),]|\\w+", program)
        return tokens

    def evaluate(self, program: str, inputs: NDArray[np.int8]) -> int:
        # Create variable namespace: x0, x1, x2, ...
        namespace: dict[str, Union[bool, Any]] = {}
        for i, val in enumerate(inputs):
            namespace[f"x{i}"] = bool(val)
        # Inject logical operators
        namespace.update(self.operations)
        try:
            # Execute expression with no builtins
            result = eval(program, {"__builtins__": {}}, namespace)
            return 1 if result else 0

        except Exception:
            return 0
