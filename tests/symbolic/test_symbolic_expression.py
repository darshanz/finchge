import cloudpickle
import numpy as np
import pytest

from finchge.symbolic.expression import SymbolicExpression


@pytest.mark.parametrize("expr", ["x0 + x1", "sin(x0) + cos(x1)"])
def test_common_expressions_evaluate(expr):
    X = np.random.default_rng(1).normal(size=(10, 5))

    result = SymbolicExpression(expr).eval(X)

    assert result.shape == (10,)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("x0 + x1", lambda X: X[:, 0] + X[:, 1]),
        ("sin(x0)", lambda X: np.sin(X[:, 0])),
        ("x2 + x7", lambda X: X[:, 2] + X[:, 7]),
    ],
)
def test_expression_values(expr, expected):
    X = np.random.default_rng(2).normal(size=(30, 10))

    result = SymbolicExpression(expr).eval(X)

    assert np.allclose(result, expected(X))


def test_missing_column_error():
    X = np.random.randn(20, 3)
    expr = SymbolicExpression("x4 + x0")
    with pytest.raises(ValueError):
        expr.eval(X)


def test_deterministic_evaluation():
    X = np.random.randn(50, 3)
    expr = SymbolicExpression("x0 * x1 + x2")
    r1 = expr.eval(X)
    r2 = expr.eval(X)
    assert np.allclose(r1, r2)


def test_constant_expression():
    """Scalar / Broadcast Behaviour"""
    X = np.random.randn(30, 5)
    expr = SymbolicExpression("5.0")
    prediction_ = expr.eval(X)
    assert np.all(prediction_ == 5)


@pytest.mark.parametrize("expr", ["", "   "])
def test_empty_expressions_are_rejected(expr):
    with pytest.raises(ValueError):
        SymbolicExpression(expr)


def test_single_sample():
    X = np.random.randn(1, 3)
    expr = SymbolicExpression("x0 + x1")
    prediction_ = expr.eval(X)
    assert prediction_.shape == (1,)


def test_log_domain():
    X = np.abs(np.random.randn(20, 1)) + 1e-6
    expr = SymbolicExpression("log(x0)")
    prediction_ = expr.eval(X)
    assert not np.any(np.isnan(prediction_))


def test_unknown_variable():
    with pytest.raises(Exception):
        SymbolicExpression("y0 + x1")


def test_complexity_metrics():
    expr = SymbolicExpression("x0 + x1 * x2")

    assert expr.node_count() > 0
    assert expr.max_depth() > 0


def test_serialization():
    expr = SymbolicExpression("sin(x0) + x1")

    blob = cloudpickle.dumps(expr)
    restored = cloudpickle.loads(blob)

    X = np.random.randn(10, 2)

    assert np.allclose(expr.eval(X), restored.eval(X))


def test_equivalent_forms():
    X = np.random.randn(50, 1)

    e1 = SymbolicExpression("x0 + x0")
    e2 = SymbolicExpression("2*x0")

    assert np.allclose(e1.eval(X), e2.eval(X))


def test_variable_order_consistency():
    e1 = SymbolicExpression("x1 + x0")
    e2 = SymbolicExpression("x0 + x1")
    assert set(map(str, e1.variables)) == set(map(str, e2.variables))


def test_protected_division_handles_zero_denominator():
    X = np.array([[2.0, 0.0], [6.0, 3.0]])

    result = SymbolicExpression("x0 / x1").eval(X)

    assert np.all(np.isfinite(result))
    assert result[0] == 1.0
    assert result[1] == 2.0


def test_numpy_column_syntax_is_supported():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = SymbolicExpression("x[:, 0] + x[:, 1]").eval(X)

    assert np.allclose(result, X[:, 0] + X[:, 1])


def test_math_constants_are_supported():
    X = np.zeros((4, 1))

    result = SymbolicExpression("pi + e").eval(X)

    assert np.allclose(result, np.pi + np.e)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "x0 if x0 else x1",
        "[x0]",
        "sin(x0, bad=1)",
        "np.sin(x0)",
    ],
)
def test_unsupported_syntax_is_rejected(expr):
    with pytest.raises(Exception):
        SymbolicExpression(expr)


def test_one_dimensional_input_is_single_sample():
    result = SymbolicExpression("x0 + x1").eval([2.0, 3.0])

    assert result.shape == (1,)
    assert result[0] == 5.0


def test_wrong_function_arity_raises_evaluation_error():
    expr = SymbolicExpression("sin(x0, x1)")
    X = np.ones((3, 2))

    with pytest.raises(Exception):
        expr.eval(X)
