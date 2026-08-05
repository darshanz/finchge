import math

import numpy as np
import pytest

from finchge.fitness.fitness_functions import (
    AccuracyFitness,
    MAEFitness,
    MSEFitness,
    RewardFitness,
    RMSEFitness,
    StringMatchFitness,
)


def test_string_match_perfect_match_returns_zero():
    fn = StringMatchFitness("hello")
    result = fn.evaluate({"phenotype": "hello"})
    assert result.value == 0


def test_string_match_complete_mismatch():
    fn = StringMatchFitness("abc")
    result = fn.evaluate({"phenotype": "xyz"})
    assert result.value == 3


def test_string_match_partial():
    fn = StringMatchFitness("hello")
    result = fn.evaluate({"phenotype": "hXllX"})
    assert result.value == 2


def test_string_match_shorter_phenotype_penalises_length():
    fn = StringMatchFitness("hello")
    result = fn.evaluate({"phenotype": "hel"})
    # 3 chars match, max_len = 5, so penalty = 5 - 3 = 2
    assert result.value == 2


def test_string_match_maximize_is_false():
    fn = StringMatchFitness("x")
    assert fn.maximize is False


def test_rmse_zero_on_perfect_predictions():
    fn = RMSEFitness()
    y = np.array([1.0, 2.0, 3.0])
    result = fn.evaluate({"y_true": y, "y_pred": y})
    assert result.value == pytest.approx(0.0)


def test_rmse_positive_on_error():
    fn = RMSEFitness()
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    result = fn.evaluate({"y_true": y_true, "y_pred": y_pred})
    assert result.value == pytest.approx(1.0)


def test_rmse_nan_pred_returns_inf():
    fn = RMSEFitness()
    y = np.array([1.0, 2.0])
    result = fn.evaluate({"y_true": y, "y_pred": np.array([float("nan"), 1.0])})
    assert math.isinf(result.value)


def test_rmse_maximize_is_false():
    assert RMSEFitness().maximize is False


def test_mse_zero_on_perfect_predictions():
    fn = MSEFitness()
    y = np.array([1.0, 2.0, 3.0])
    result = fn.evaluate({"y_true": y, "y_pred": y})
    assert result.value == pytest.approx(0.0)


def test_mse_squared_error():
    fn = MSEFitness()
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([2.0, 2.0])
    result = fn.evaluate({"y_true": y_true, "y_pred": y_pred})
    assert result.value == pytest.approx(4.0)


def test_mse_nan_pred_returns_inf():
    fn = MSEFitness()
    result = fn.evaluate(
        {"y_true": np.array([1.0]), "y_pred": np.array([float("nan")])}
    )
    assert math.isinf(result.value)


def test_mae_zero_on_perfect_predictions():
    fn = MAEFitness()
    y = np.array([1.0, 2.0])
    result = fn.evaluate({"y_true": y, "y_pred": y})
    assert result.value == pytest.approx(0.0)


def test_mae_absolute_error():
    fn = MAEFitness()
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 1.0, 2.0])
    result = fn.evaluate({"y_true": y_true, "y_pred": y_pred})
    assert result.value == pytest.approx(2.0)


def test_mae_shape_mismatch_raises():
    fn = MAEFitness()
    with pytest.raises(ValueError, match="Shape mismatch"):
        fn.evaluate({"y_true": np.array([1.0, 2.0]), "y_pred": np.array([1.0])})


# --- RewardFitness ---


def test_reward_fitness_sums_rewards():
    fn = RewardFitness()
    result = fn.evaluate({"y_pred": np.array([10.0, 20.0, 30.0])})
    assert result.value == pytest.approx(60.0)


def test_reward_fitness_maximize_is_true():
    assert RewardFitness().maximize is True


def test_accuracy_perfect():
    fn = AccuracyFitness()
    y = np.array([0, 1, 1, 0])
    result = fn.evaluate({"y_true": y, "y_pred": y})
    assert result.value == pytest.approx(1.0)


def test_accuracy_half_correct():
    fn = AccuracyFitness()
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    result = fn.evaluate({"y_true": y_true, "y_pred": y_pred})
    assert result.value == pytest.approx(0.5)


def test_accuracy_maximize_is_true():
    assert AccuracyFitness().maximize is True
