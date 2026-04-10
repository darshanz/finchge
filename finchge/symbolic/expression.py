from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from sympy.utilities.lambdify import lambdify


class SymbolicExpression:
    """
    Represents a symbolic mathematical expression for evaluation and analysis.

    This class provides a lightweight interface for parsing,
    validating, evaluating, and analysing symbolic expressions used in
    symbolic regression and genetic programming workflows.

    Expressions are parsed using SymPy and evaluated numerically using
    vectorised NumPy functions generated via `sympy.lambdify`.

    The evaluator supports:
        - Multiple input variables (e.g., x0, x1, x2, ... or in numpy style column slice syntax e.g. x[: 0])
        - Custom symbolic functions
        - Constant-only expressions
        - Partial column selection from input data
        - Structural complexity analysis
        - Optional symbolic simplification

    The class intentionally avoids automatic simplification during evaluation
    to ensure predictable runtime performance in evolutionary algorithms.

    Attributes:
        expression: Original expression string (after slice-pattern conversion).
        expr: Parsed SymPy expression object.
        symbols: Ordered tuple of SymPy symbols used in the expression.
        functions: Dictionary mapping function names to symbolic implementations.
    """

    def __init__(
        self,
        expression: str,
        variables: Optional[Iterable[str]] = None,
        functions: Optional[Dict[str, Callable[..., object]]] = None,
        safe_mode: bool = True,
    ) -> None:
        """
        Initializes a symbolic expression.

        Args:
            expression: String representation of a mathematical expression.
                Expressions may reference variables using the format ``x0``, ``x1``,
                etc. Slice-array syntax such as ``x[:, 3]`` is automatically converted
                to symbolic form.

            variables: Optional iterable specifying allowed variable names.
                If provided, variables are explicitly registered. Otherwise,
                variables are inferred automatically from the parsed expression.

            functions: Optional dictionary mapping function names to symbolic
                implementations. If not provided, a default set of mathematical
                functions is used.
            safe_mode : enable numerical safety

        Raises:
            ValueError: If the expression is empty or contains unsupported variables.
            SympifyError: If the expression cannot be parsed into a symbolic form.

        """
        self.safe_mode = safe_mode
        # empty strings or whitespaces should not be accepted
        # raise error if any
        if expression is None or not str(expression).strip():
            raise ValueError("Expression string is empty.")

        # Slice array pattern is also supported by FinchGE parser, however sympy style notation is used internally.
        self.original_expression = expression
        self.expression = self._replace_slice_array_pattern(expression)

        # SYMBOLIC functions (for parsing)
        self.symbolic_functions = functions or {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "sign": sp.sign,
            "floor": sp.floor,
            "ceil": sp.ceiling,
            "erf": sp.erf,
            "gamma": sp.gamma,
            "digamma": sp.digamma,
            "aq": self._analytic_quotient,
        }

        # NUMERICAL functions for lambdify, provide safety
        self.numerical_modules = self._create_numerical_modules()

        # Parse expression  with TimeOut
        self._parse_expression(variables)

        self._validate_symbols()

        # Numerical evaluator - using Callable type instead of BenchmarkCallable
        self._numeric_func: Optional[Callable[..., NDArray[np.float64]]] = None

    def _create_numerical_modules(self) -> List[Union[Dict[str, Any], str]]:
        """
        safe numerical implementations.
        """
        safe_numpy: Dict[str, Any] = {
            "add": lambda a, b: a + b,
            "mul": lambda a, b: a * b,
            "sub": lambda a, b: a - b,
            "div": self._safe_div,
            "/": self._safe_div,
            "truediv": self._safe_div,
            "pow": self._safe_pow,
            "Pow": self._safe_pow,
            "exp": self._safe_exp,
            "log": self._safe_log,
            "ln": self._safe_log,
            "tan": self._safe_tan,
            "cot": self._safe_cot,
            "sec": self._safe_sec,
            "csc": self._safe_csc,
            "cosh": self._safe_cosh,
            "sinh": np.sinh,
            "tanh": np.tanh,
            "asin": self._safe_asin,
            "acos": self._safe_acos,
            "atan": np.arctan,
            "sqrt": self._safe_sqrt,
            "sin": self._safe_sin,
            "cos": self._safe_cos,
            "abs": np.abs,
            "sign": np.sign,
            "pdiv": self._safe_div,
            "psqrt": self._safe_sqrt,
            "plog": self._safe_log,
        }

        return [safe_numpy, "numpy"]

    def _parse_expression(self, variables: Optional[Iterable[str]]) -> None:
        """
        Parse the expression into a SymPy object safely.
        """
        # Temporary symbols for parsing
        local_dict = dict(self.symbolic_functions)
        try:
            if variables is not None:
                self.symbols = tuple(sp.Symbol(v) for v in variables)
                local_dict.update({v: s for v, s in zip(variables, self.symbols)})
                self.expr = self.safe_sympify(self.expression, local_dict=local_dict)
            else:
                # Infer variables from expression
                self.expr = self.safe_sympify(self.expression, local_dict=local_dict)
                self.symbols = tuple(sorted(self.expr.free_symbols, key=str))
        except (RecursionError, RuntimeError, ValueError) as e:
            raise ValueError(
                f"Failed to parse expression: {e}. Expression: {self.expression[:100]}"
            )

        if hasattr(self.expr, "doit"):
            # avoid doit() :: it can trigger evaluation
            pass

    def safe_sympify(
        self,
        expression: str,
        local_dict: Optional[Dict[str, Any]] = None,
        max_length: int = 200,
    ) -> sp.Expr:
        """Safe sympify with length limit."""

        #  Hard limit on expression length Before the max length
        if len(expression) > 500:  # Hard limit :: max_length is checked later
            raise ValueError(f"Expression too long: {len(expression)} > 500")

        # numeric constants that are too large
        import re

        large_numbers = re.findall(
            r"\b\d{6,}\b", expression
        )  # Numbers with more than 6 digits
        if large_numbers:
            raise ValueError(f"Expression contains huge constants: {large_numbers[:3]}")

        # patterns like cos(999999)
        huge_trig_args = re.findall(r"(sin|cos|tan)\((\d{6,})\)", expression)
        if huge_trig_args:
            raise ValueError(f"Huge constant in trig function: {huge_trig_args[0]}")

        if len(expression) > max_length:
            raise ValueError(f"Expression too long: {len(expression)} > {max_length}")

        trig_count = sum(expression.count(func) for func in ["sin", "cos", "tan"])
        if trig_count > 15:  # Adjust threshold as needed
            raise ValueError(f"Too many trig functions: {trig_count}")

        # avoiding nested trig patterns
        if (
            "sin(sin" in expression
            or "cos(cos" in expression
            or "tan(tan" in expression
        ):
            raise ValueError("Deeply nested trig functions detected")

        # avoiding nested exp explosions
        if expression.count("exp") > 2:
            raise ValueError("Too many nested exp functions")

        return sp.sympify(expression, locals=local_dict, evaluate=False)

    def _replace_slice_array_pattern(self, expr_with_slice_notation: str) -> str:
        """
        Replace x[:, N] with sympy style xN
        if the phenotype has expr_with_slice_notation
        """
        pattern = r"x\[:,\s*(\d+)\]"
        return re.sub(pattern, r"x\1", expr_with_slice_notation)

    def _validate_symbols(self) -> None:
        """Ensure all symbols follow the x<i> naming convention."""
        for sym in self.expr.free_symbols:
            name = str(sym)
            if not name.startswith("x"):
                raise ValueError(f"Invalid variable name: {name}")
            try:
                int(name[1:])
            except ValueError:
                raise ValueError(f"Invalid variable name: {name}")

    def __str__(self) -> str:
        """Returns a human-readable string representation of the expression.

        Returns:
            String form of the underlying SymPy expression.
        """
        return str(self.expr)

    def __repr__(self) -> str:
        """
        Returns an unambiguous representation of the symbolic expression.

        Returns:
            String representation suitable for debugging and logging.
        """
        return f"SymbolicExpression({self.expr})"

    def _repr_html_(self) -> str:
        """
        Provides HTML representation for Jupyter notebook display.

        Returns:
            str: LaTeX-formatted expression enclosed in math delimiters for node<50
        """
        nodes = self.node_count(simplify=False)
        depth = self.max_depth()

        summary = f"<b>Nodes:</b> {nodes} &nbsp;&nbsp; " f"<b>Depth:</b> {depth}"

        # long expressions do not look good when displayed as LaTeX
        # So only short expressions (50 chosen arbitrarily) in LaTeX
        # show string form for larger ones.
        if nodes <= 50:
            return summary + "<br>" + f"$$ {sp.latex(self.expr)} $$"

        full_expr = str(self.expr)

        return f"""
        {summary}
        <br>
        <details>
            <summary><b>Show Full Expression</b></summary>
            <pre style="white-space: pre-wrap; word-break: break-word;">
            {full_expr}
            </pre>
        </details>
        """

    def simplify(self) -> "SymbolicExpression":
        """Returns a simplified symbolic expression.

        Simplification is performed using SymPy trigonometric simplification.
        A new ``SymbolicExpression`` instance is returned, preserving variable
        and function definitions.

        Returns:
            SymbolicExpression: New instance containing the simplified expression.

        Note:
            Simplification is intended primarily for display, reporting, and
            post-evolution analysis. It is not required for numerical evaluation.
        """
        simplified = sp.trigsimp(str(self.expr))
        return SymbolicExpression(
            str(simplified),
            variables=[str(v) for v in self.symbols],
            functions=self.symbolic_functions,
        )

    def eval(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates the symbolic expression numerically with safety protections.
        """
        # LARGE_NUM_FILLER = 10e10
        LARGE_NUM_FILLER = np.inf
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples = X_arr.shape[0]

        with np.errstate(all="ignore"):
            # constant only expressions
            if not self.symbols:
                if self._numeric_func is None:
                    # For constant expressions, lambdify with empty symbols list
                    self._numeric_func = lambdify(
                        [],  # Empty list for no arguments
                        self.expr,
                        modules=self.numerical_modules if self.safe_mode else ["numpy"],
                        dummify=False,
                    )
                try:
                    value = self._numeric_func()  # no args
                except Exception as e:
                    logging.error(f"Exception in worker: {e}", exc_info=True)
                    # Fallback to large constant on error
                    return np.full(n_samples, LARGE_NUM_FILLER, dtype=np.float64)

                # Broadcast scalar to (n_samples,)
                return np.full(n_samples, float(np.asarray(value)), dtype=np.float64)

            # get required column indices from variables
            required_indices = [self._symbol_to_index(sym) for sym in self.symbols]
            max_required = max(required_indices)

            if X_arr.shape[1] <= max_required:
                raise ValueError(
                    f"Expression requires column x{max_required}, "
                    f"but input only has {X_arr.shape[1]} columns."
                )

            # Extract only relevant columns in correct order
            selected_columns = [X_arr[:, i] for i in required_indices]

            # Lazy compilation
            if self._numeric_func is None:
                self._numeric_func = lambdify(
                    self.symbols,
                    self.expr,
                    modules=self.numerical_modules if self.safe_mode else ["numpy"],
                    dummify=False,
                )

            try:
                result = self._numeric_func(*selected_columns)

                # this line  sometimes triggers the TypeError.
                # Perhaps if the lambdified functions returns a complex result,
                # So, using .real on the result of the numeric function call.
                if isinstance(result, np.ndarray) and np.iscomplexobj(result):
                    result = result.real
                elif isinstance(result, complex):
                    result = result.real

                result_arr = np.asarray(result, dtype=np.float64)

                # Final safety check for NaN
                if np.any(np.isnan(result_arr)):
                    result_arr = np.where(
                        np.isnan(result_arr), LARGE_NUM_FILLER, result_arr
                    )

                return result_arr
            except RecursionError as e:  # recursion protection
                logging.error(f"Recursion error in evaluation: {e}")
                return np.full(n_samples, LARGE_NUM_FILLER, dtype=np.float64)

            except Exception as e:
                logging.error(f"Exception in worker: {e}", exc_info=True)

                # Return large constant on any evaluation error
                return np.full(n_samples, LARGE_NUM_FILLER, dtype=np.float64)

    def _symbol_to_index(self, sym: sp.Symbol) -> int:
        # get required column indices from variables
        name = str(sym)
        if not name.startswith("x"):
            raise ValueError(f"Unsupported variable naming: {name}")
        try:
            return int(name[1:])
        except ValueError:
            raise ValueError(f"Unsupported variable naming: {name}")

    def node_count(self, simplify: bool = True) -> int:
        """
        Computes the number of nodes in the symbolic expression tree.

        This metric is commonly used as a measure of expression complexity
        in symbolic regression.

        Args:
            simplify: Whether to apply trigonometric simplification before
                computing node count.

        Returns:
            int: Total number of nodes in the expression tree.
        """
        expr = sp.trigsimp(str(self.expr)) if simplify else self.expr
        return sum(1 for _ in sp.preorder_traversal(expr))

    def max_depth(self) -> int:
        """
        Computes the depth of the symbolic expression tree.

        Tree depth corresponds to the maximum nesting level of operations
        within the expression and is commonly used as a structural complexity
        metric.

        Returns:
            int: Maximum depth of the expression tree.
        """

        def _depth(node: sp.Basic) -> int:
            if not node.args:
                return 1
            return 1 + max(_depth(a) for a in node.args)

        return _depth(self.expr)

    @property
    def variables(self) -> Tuple[sp.Symbol, ...]:
        """
        Returns variables used in the expression.

        Returns:
            Tuple of SymPy symbols appearing in the expression. Variables are
            returned in deterministic sorted order.
        """
        return self.symbols

    # SafeMath methods
    @staticmethod
    def _safe_div(
        a: Union[float, NDArray[np.float64]], b: Union[float, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Safe division."""
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)

        # Handle zero/almost zero denominators
        safe_b: NDArray[np.float64] = np.where(np.abs(b_arr) < 1e-10, 1.0, b_arr)
        result: NDArray[np.float64] = np.asarray(a_arr / safe_b, dtype=np.float64)

        signs = np.where(np.sign(result) == 0, 1.0, np.sign(result))
        result = np.where(np.isfinite(result), result, 1e10 * signs)
        return result

    @staticmethod
    def _safe_pow(
        a: Union[float, NDArray[np.float64]], b: Union[float, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Safe power operation."""
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)

        # np.abs --- to prevent complex results from negative bases
        result: NDArray[np.float64] = np.power(np.abs(a_arr), b_arr)

        # If the result becomes complex, just real part should be used
        if np.iscomplexobj(result):
            result = result.real

        # Preserve sign for integer exponents
        if np.ndim(b_arr) == 0:
            b_scalar = float(b_arr)
            if b_scalar.is_integer():
                b_int = int(b_scalar)
                odd_mask = (a_arr < 0) & (b_int % 2 == 1)
                result = np.where(odd_mask, -result, result)
        else:
            b_int_mask = np.mod(b_arr, 1) == 0
            odd_negative_mask = (a_arr < 0) & b_int_mask & (np.mod(b_arr, 2) == 1)
            result = np.where(odd_negative_mask, -result, result)

        result = np.where(np.isfinite(result), result, 1e10)
        return result

    @staticmethod
    def _safe_exp(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe exponential."""
        # if x is compplex object just take the real part.
        if np.iscomplexobj(x):
            x = x.real
        elif isinstance(x, complex):
            x = x.real

        x_arr = np.asarray(x, dtype=np.float64)

        clipped: NDArray[np.float64] = np.clip(x_arr, -700, 700)
        return np.asarray(np.exp(clipped), dtype=np.float64)

    @staticmethod
    def _safe_log(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe logarithm."""

        x_arr = np.asarray(x, dtype=np.float64)
        return np.asarray(np.log(np.abs(x_arr) + 1e-10), dtype=np.float64)

    @staticmethod
    def _safe_tan(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe tangent."""
        x_arr = np.asarray(x, dtype=np.float64)
        near_pole: NDArray[np.bool_] = np.abs(np.mod(x_arr - np.pi / 2, np.pi)) < 0.1
        x_shifted: NDArray[np.float64] = np.where(near_pole, x_arr + 0.1, x_arr)

        result: NDArray[np.float64] = np.tan(x_shifted)
        result = np.where(np.isfinite(result), result, 0.0)
        return result

    @staticmethod
    def _safe_cos(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe cosine with argument reduction to prevent recursion."""
        x_arr = np.asarray(x, dtype=np.float64)
        # Reduce large arguments modulo 2*PI to prevent SymPy recursion
        x_arr = np.where(np.abs(x_arr) > 1e6, np.mod(x_arr, 2 * np.pi), x_arr)
        result = np.cos(x_arr)
        return np.where(np.isfinite(result), result, 0.0)

    @staticmethod
    def _safe_sin(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe sine with argument reduction to prevent recursion."""
        x_arr = np.asarray(x, dtype=np.float64)
        # Reduce large arguments modulo 2* PI to prevent SymPy recursion
        x_arr = np.where(np.abs(x_arr) > 1e6, np.mod(x_arr, 2 * np.pi), x_arr)
        result = np.sin(x_arr)
        return np.where(np.isfinite(result), result, 0.0)

    @staticmethod
    def _safe_cot(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe cotangent."""
        x_arr = np.asarray(x, dtype=np.float64)
        tan_x = np.tan(x_arr)
        result = np.where(np.abs(tan_x) < 1e-10, 1e10, 1.0 / tan_x)
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _safe_sec(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe secant."""
        x_arr = np.asarray(x, dtype=np.float64)
        cos_x: NDArray[np.float64] = np.cos(x_arr)

        result: NDArray[np.float64] = np.where(np.abs(cos_x) < 1e-10, 1e10, 1.0 / cos_x)
        return result

    @staticmethod
    def _safe_csc(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe cosecant."""
        x_arr = np.asarray(x, dtype=np.float64)
        sin_x: NDArray[np.float64] = np.sin(x_arr)

        result: NDArray[np.float64] = np.where(np.abs(sin_x) < 1e-10, 1e10, 1.0 / sin_x)
        return result

    @staticmethod
    def _safe_cosh(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe hyperbolic cosine."""
        x_arr = np.asarray(x, dtype=np.float64)
        clipped: NDArray[np.float64] = np.clip(x_arr, -50, 50)
        return np.asarray(np.cosh(clipped), dtype=np.float64)

    @staticmethod
    def _safe_sqrt(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe square root."""
        x_arr = np.asarray(x, dtype=np.float64)
        return np.asarray(np.sqrt(np.abs(x_arr)), dtype=np.float64)

    @staticmethod
    def _safe_asin(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe arcsin."""
        x_arr = np.asarray(x, dtype=np.float64)
        clipped: NDArray[np.float64] = np.clip(x_arr, -1, 1)
        return np.asarray(np.arcsin(clipped), dtype=np.float64)

    @staticmethod
    def _safe_acos(x: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Safe arccos."""
        x_arr = np.asarray(x, dtype=np.float64)
        clipped: NDArray[np.float64] = np.clip(x_arr, -1, 1)
        return np.asarray(np.arccos(clipped), dtype=np.float64)

    def _analytic_quotient(
        self,
        a: Union[float, NDArray[np.float64]],
        b: Union[float, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Analytic quotient for symbolic differentiation."""
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        # Use numpy operations for numeric evaluation
        return np.asarray(a_arr / np.sqrt(1.0 + b_arr**2.0), dtype=np.float64)
