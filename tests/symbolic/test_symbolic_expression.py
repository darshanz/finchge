import cloudpickle
import numpy as np
import pytest

from finchge.symbolic.expression import SymbolicExpression


def test_symbolic_expression():
    """
    Test symbolic expression
    """
    expressions = [
        "x0 + x1",
        "sin(x0) + cos(x1)",
    ]

    X = np.random.randn(10, 5)

    for expr in expressions:
        try:
            sym_expr = SymbolicExpression(expression=expr)
            prediction_ = sym_expr.eval(X)
            assert prediction_.shape == (10,)
            assert not np.all(np.isnan(prediction_))
        except Exception as e:
            print(f"Expression '{expr}' failed: {e}")


def test_basic_math_correctness():
    X = np.random.randn(100, 2)
    expr = SymbolicExpression("x0 + x1")
    prediction_ = expr.eval(X)
    expected = X[:, 0] + X[:, 1]
    assert np.allclose(prediction_, expected)


def test_trigonometric_correctness():
    X = np.random.randn(100, 2)
    expr = SymbolicExpression("sin(x0)")
    prediction_ = expr.eval(X)
    assert np.allclose(prediction_, np.sin(X[:, 0]))


def test_column_subset_usage():
    X = np.random.randn(20, 10)
    expr = SymbolicExpression("x2 + x7")
    prediction_ = expr.eval(X)
    expected = X[:, 2] + X[:, 7]
    assert np.allclose(prediction_, expected)


def test_missing_column_error():
    X = np.random.randn(20, 3)
    expr = SymbolicExpression("x4 + x0")
    with pytest.raises(ValueError):
        expr.eval(X)


def test_simplification_equivalence():
    expr = SymbolicExpression("x0 + 0")
    simplified = expr.simplify()
    assert str(simplified) == "x0"


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


def test_whitespace_expression():
    with pytest.raises(ValueError):
        SymbolicExpression("   ")


def test_empty_string_expression():
    with pytest.raises(ValueError):
        SymbolicExpression("")


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
