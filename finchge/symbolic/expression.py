from __future__ import annotations

import ast
import re
from typing import Any, Callable, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from finchge.symbolic.protected_math import pmath

FloatArray = NDArray[np.float64]


class ExpressionSyntaxError(ValueError):
    """Raised when an expression uses unsupported syntax."""


class ExpressionEvaluationError(ValueError):
    """Raised when an expression cannot be evaluated."""


class SymbolicExpression:
    """
    Parse and evaluate a restricted symbolic expression.

    The class provides a small, controlled expression language intended for
    symbolic regression and genetic programming. Expressions are parsed using
    Python's AST and evaluated using protected numerical operators to avoid
    NaN/Inf where possible.

    Supported features:
        - Variables of the form x0, x1, x2, ...
        - Basic arithmetic operations (+, -, *, /, **)
        - Selected mathematical functions (sin, cos, exp, log, etc.)
        - Constant expressions
        - Optional normalization of NumPy-style column syntax (e.g. x[:, 0] → x0)
        - Simple structural metrics (node count, depth)

    Notes:
        This is not a full symbolic algebra system. Expressions are evaluated
        numerically and are not automatically simplified.

    Attributes:
        original_expression (str):
            The input expression string as provided by the user.

        expression (str):
            Normalized expression string after preprocessing (e.g. slice syntax rewritten).

        variables (list[str]):
            Sorted list of variable names used in the expression (e.g. ['x0', 'x2']).

        _tree (ast.Expression):
            Parsed AST representation of the expression.

    Raises:
        ExpressionSyntaxError:
            If the expression contains unsupported syntax.

        ExpressionEvaluationError:
            If evaluation fails at runtime.

    """

    _SLICE_VAR_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"x\s*\[\s*:\s*,\s*(\d+)\s*\]"
    )
    _VAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bx(\d+)\b")

    _BINARY_OPS: Final[dict[type[ast.operator], Callable[[Any, Any], Any]]] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: pmath.div,
        ast.Pow: pmath.pow,
    }

    _UNARY_OPS: Final[dict[type[ast.unaryop], Callable[[Any], Any]]] = {
        ast.UAdd: lambda x: x,
        ast.USub: lambda x: -x,
    }

    _CONSTANTS: Final[dict[str, float]] = {
        "pi": float(np.pi),
        "e": float(np.e),
    }

    _FUNCTIONS: Final[dict[str, Callable[..., Any]]] = {
        "sin": pmath.sin,
        "cos": pmath.cos,
        "tan": pmath.tan,
        "asin": pmath.asin,
        "acos": pmath.acos,
        "atan": pmath.atan,
        "sinh": pmath.sinh,
        "cosh": pmath.cosh,
        "tanh": pmath.tanh,
        "asinh": pmath.asinh,
        "acosh": pmath.acosh,
        "atanh": pmath.atanh,
        "exp": pmath.exp,
        "log": pmath.log,
        "plog": pmath.log,
        "ln": pmath.log,
        "log1p_abs": pmath.log1p_abs,
        "sqrt": pmath.sqrt,
        "psqrt": pmath.psqrt,
        "abs": pmath.abs,
        "sign": pmath.sign,
        "aq": pmath.aq,
        "add": pmath.add,
        "sub": pmath.sub,
        "mul": pmath.mul,
        "div": pmath.div,
        "pdiv": pmath.div,
        "pow": pmath.pow,
    }

    _ALLOWED_AST_NODES: Final[tuple[type[ast.AST], ...]] = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    )

    def __init__(self, expression: str) -> None:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a non-empty string")

        self.original_expression = expression
        self.expression = self._normalize_expression(expression)
        self.variables = self._extract_variables(self.expression)

        try:
            parsed = ast.parse(self.expression, mode="eval")
        except SyntaxError as exc:
            raise ExpressionSyntaxError(f"invalid expression: {expression!r}") from exc

        self._validate_ast(parsed)
        self._tree = parsed

    @classmethod
    def _normalize_expression(cls, expression: str) -> str:
        # Accept x[:, 3] and normalize it to x3. support for ponyge-style gramamrs
        return cls._SLICE_VAR_PATTERN.sub(r"x\1", expression.strip())

    @classmethod
    def _extract_variables(cls, expression: str) -> list[str]:
        matches = cls._VAR_PATTERN.findall(expression)
        return sorted({f"x{m}" for m in matches}, key=lambda name: int(name[1:]))

    @classmethod
    def _validate_ast(cls, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, cls._ALLOWED_AST_NODES):
                raise ExpressionSyntaxError(
                    f"unsupported syntax: {type(node).__name__}"
                )

            if isinstance(node, ast.Name):
                name = node.id
                if not (
                    name in cls._FUNCTIONS
                    or name in cls._CONSTANTS
                    or cls._VAR_PATTERN.fullmatch(name)
                ):
                    raise ExpressionSyntaxError(f"unknown name: {name!r}")

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ExpressionSyntaxError(
                        "only direct function calls are allowed"
                    )

                func_name = node.func.id
                if func_name not in cls._FUNCTIONS:
                    raise ExpressionSyntaxError(f"unknown function: {func_name!r}")

                if node.keywords:
                    raise ExpressionSyntaxError("keyword arguments are not supported")

    def _eval_node(self, node: ast.AST, env: dict[str, Any]) -> Any:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, env)

        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, (int, float)):
                return float(value)
            raise ExpressionSyntaxError(f"unsupported constant: {value!r}")

        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self._CONSTANTS:
                return self._CONSTANTS[node.id]
            raise ExpressionSyntaxError(f"unknown name: {node.id!r}")

        if isinstance(node, ast.UnaryOp):
            unary_op_type = type(node.op)
            if unary_op_type not in self._UNARY_OPS:
                raise ExpressionSyntaxError(
                    f"unsupported unary operator: {unary_op_type.__name__}"
                )
            operand = self._eval_node(node.operand, env)
            return self._UNARY_OPS[unary_op_type](operand)

        if isinstance(node, ast.BinOp):
            binary_op_type = type(node.op)
            if binary_op_type not in self._BINARY_OPS:
                raise ExpressionSyntaxError(
                    f"unsupported binary operator: {binary_op_type.__name__}"
                )
            left = self._eval_node(node.left, env)
            right = self._eval_node(node.right, env)
            return self._BINARY_OPS[binary_op_type](left, right)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionSyntaxError("only direct function calls are allowed")

            func_name = node.func.id
            func = self._FUNCTIONS[func_name]
            args = [self._eval_node(arg, env) for arg in node.args]
            return func(*args)

        raise ExpressionSyntaxError(f"unsupported node: {type(node).__name__}")

    @staticmethod
    def _coerce_input(X: ArrayLike) -> FloatArray:
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 0:
            return X_arr.reshape(1, 1)

        if X_arr.ndim == 1:
            return X_arr.reshape(1, -1)

        if X_arr.ndim != 2:
            raise ValueError("X must be a scalar, 1D array, or 2D array")

        return X_arr

    @staticmethod
    def _coerce_output(value: Any, n_samples: int) -> FloatArray:
        arr = np.asarray(value, dtype=np.float64)

        if arr.ndim == 0:
            return np.full(n_samples, float(arr), dtype=np.float64)

        if arr.ndim == 1:
            if arr.shape[0] == n_samples:
                return np.nan_to_num(arr, nan=np.inf, posinf=np.inf, neginf=-np.inf)

            if arr.shape[0] == 1:
                return np.full(n_samples, float(arr[0]), dtype=np.float64)

        raise ExpressionEvaluationError(
            f"expression returned shape {arr.shape}, expected scalar or ({n_samples},)"
        )

    def eval(self, X: ArrayLike) -> FloatArray:
        """
        Evaluate the expression on X.

        Parameters
        ----------
        X:
            Input data. Expected shape is (n_samples, n_features).
            A scalar is treated as shape (1, 1).
            A 1D array is treated as a single sample with multiple features.

        Returns
        -------
        np.ndarray
            Shape (n_samples,).
        """
        X_arr = self._coerce_input(X)
        n_samples, n_features = X_arr.shape

        env: dict[str, Any] = dict(self._FUNCTIONS)
        env.update(self._CONSTANTS)

        for var in self.variables:
            idx = int(var[1:])
            if idx >= n_features:
                raise ValueError(
                    f"expression requires {var}, but X has only {n_features} feature(s)"
                )
            env[var] = X_arr[:, idx]

        try:
            with np.errstate(all="ignore"):
                value = self._eval_node(self._tree, env)
            return self._coerce_output(value, n_samples)

        except ExpressionSyntaxError:
            raise
        except Exception as exc:
            raise ExpressionEvaluationError(
                f"failed to evaluate expression {self.expression!r}"
            ) from exc

    def node_count(self) -> int:
        # Count actual expression nodes, not regex matches.
        count = 0
        for node in ast.walk(self._tree):
            if isinstance(
                node, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)
            ):
                count += 1
        return count

    def max_depth(self) -> int:
        def depth(node: ast.AST) -> int:
            children = list(ast.iter_child_nodes(node))
            if not children:
                return 1
            return 1 + max(depth(child) for child in children)

        # Skip the outer ast.Expression wrapper.
        return max(1, depth(self._tree.body))

    def __str__(self) -> str:
        return self.expression

    def __repr__(self) -> str:
        return f"SymbolicExpression({self.expression!r})"
