from typing import Any, Optional

import numpy as np
import pytest

from finchge.runners import (
    DirectEvalRunner,
    MLModelRunner,
    SymbolicRegressionRunner,
    TrainEvalRunner,
)


class DummyDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class DummyDirectRunner(DirectEvalRunner):
    def predict_direct(self, phenotype: str, X_eval: np.ndarray) -> np.ndarray:
        if phenotype == "raise":
            raise RuntimeError("boom")
        return np.sum(X_eval, axis=1)


class DummyModel:
    def __init__(self):
        self.fit_called = False
        self.fit_args = None
        self.feature_importances_ = np.array([0.7, 0.3])

    def fit(self, X, y):
        self.fit_called = True
        self.fit_args = (X, y)

    def predict(self, X):
        return np.sum(X, axis=1)

    def predict_proba(self, X):
        # simple fake binary probabilities
        s = np.clip(np.sum(X, axis=1), 0.0, 1.0)
        return np.column_stack([1.0 - s, s])

    def decision_function(self, X):
        return np.sum(X, axis=1)


class DummyTorchModel:
    def __init__(self):
        self.fit_called = False
        self.train_dataset = None
        self.val_dataset = None

    def fit(self, train_dataset=None, val_dataset=None):
        self.fit_called = True
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def predict(self, dataset):
        X = []
        y = []
        for i in range(len(dataset)):
            xi, yi = dataset[i]
            X.append(np.array(xi, dtype=np.float64))
            y.append(float(yi))
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        y_pred = np.sum(X, axis=1)
        return y_pred, y


class DummyModelParser:
    def parse(self, phenotype: str):
        if phenotype == "torch_model":
            return DummyTorchModel()
        return DummyModel()


class DummyTrainEvalRunner(TrainEvalRunner):
    def build_model(self, phenotype: str) -> Any:
        if phenotype == "bad_model":
            raise RuntimeError("cannot build")
        return DummyModel()

    def fit_model(
        self,
        model: Any,
        X_train: Optional[np.ndarray],
        y_train: Optional[np.ndarray],
    ) -> Any:
        if X_train is None or y_train is None:
            raise ValueError("missing training data")
        model.fit(X_train, y_train)
        return model

    def predict_eval(
        self,
        model: Any,
        X_eval: Optional[np.ndarray],
    ) -> np.ndarray:
        if X_eval is None:
            raise ValueError("missing eval data")
        return model.predict(X_eval)

    def predict_eval_proba(
        self,
        model: Any,
        X_eval: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if X_eval is None:
            return None
        return model.predict_proba(X_eval)

    def predict_eval_score(
        self,
        model: Any,
        X_eval: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if X_eval is None:
            return None
        return model.decision_function(X_eval)


@pytest.fixture
def split_data():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_train = np.array([10.0, 20.0])

    X_val = np.array([[5.0, 6.0], [7.0, 8.0]])
    y_val = np.array([30.0, 40.0])

    X_test = np.array([[9.0, 10.0], [11.0, 12.0]])
    y_test = np.array([50.0, 60.0])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


@pytest.fixture
def dataset_splits():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_train = np.array([10.0, 20.0])

    X_val = np.array([[5.0, 6.0], [7.0, 8.0]])
    y_val = np.array([30.0, 40.0])

    X_test = np.array([[9.0, 10.0], [11.0, 12.0]])
    y_test = np.array([50.0, 60.0])

    return (
        DummyDataset(X_train, y_train),
        DummyDataset(X_val, y_val),
        DummyDataset(X_test, y_test),
    )


# DataAwareRunner


def test_data_aware_runner_defaults_eval_split_to_val(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)
    assert runner.eval_split == "val"


def test_data_aware_runner_set_eval_split_changes_active_split(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)

    runner.set_eval_split("train")
    assert runner.eval_split == "train"

    runner.set_eval_split("test")
    assert runner.eval_split == "test"


def test_data_aware_runner_rejects_invalid_split(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)

    with pytest.raises(ValueError, match="train|val|test"):
        runner.set_eval_split("dev")


def test_data_aware_runner_infers_tuple_numpy_type(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)

    assert runner.data_info["format"] == "tuple"
    assert runner.data_info["X_type"] == "numpy"


def test_data_aware_runner_infers_dataset_torch_type(dataset_splits):
    train, val, test = dataset_splits
    runner = DummyDirectRunner(train, val, test)

    assert runner.data_info["format"] == "dataset"
    assert runner.data_info["X_type"] == "torch"


def test_data_aware_runner_get_split_data_returns_requested_split(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)

    X_train, y_train = runner.get_split_data("train")
    np.testing.assert_array_equal(X_train, train[0])
    np.testing.assert_array_equal(y_train, train[1])

    X_val, y_val = runner.get_split_data("val")
    np.testing.assert_array_equal(X_val, val[0])
    np.testing.assert_array_equal(y_val, val[1])


def test_data_aware_runner_get_eval_data_returns_active_split(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)

    runner.set_eval_split("test")
    X_eval, y_eval = runner.get_eval_data()

    np.testing.assert_array_equal(X_eval, test[0])
    np.testing.assert_array_equal(y_eval, test[1])


# DirectEvalRunner


def test_direct_eval_runner_run_returns_required_base_context(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)

    context = runner.run("any_phenotype")

    assert set(context.keys()) == {"phenotype", "y_true", "y_pred"}
    assert context["phenotype"] == "any_phenotype"
    np.testing.assert_array_equal(context["y_true"], val[1])
    np.testing.assert_array_equal(context["y_pred"], np.sum(val[0], axis=1))


def test_direct_eval_runner_uses_active_eval_split(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)
    runner.set_eval_split("train")

    context = runner.run("any_phenotype")

    np.testing.assert_array_equal(context["y_true"], train[1])
    np.testing.assert_array_equal(context["y_pred"], np.sum(train[0], axis=1))


def test_direct_eval_runner_adds_requested_eval_keys(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)

    context = runner.run("any_phenotype", context_hints={"X", "X_eval", "y_eval"})

    assert "X" in context
    assert "X_eval" in context
    assert "y_eval" in context
    np.testing.assert_array_equal(context["X"], val[0])
    np.testing.assert_array_equal(context["X_eval"], val[0])
    np.testing.assert_array_equal(context["y_eval"], val[1])


def test_direct_eval_runner_adds_requested_split_specific_keys(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)

    context = runner.run(
        "any_phenotype",
        context_hints={"X_train", "y_train", "X_test", "y_test"},
    )

    np.testing.assert_array_equal(context["X_train"], train[0])
    np.testing.assert_array_equal(context["y_train"], train[1])
    np.testing.assert_array_equal(context["X_test"], test[0])
    np.testing.assert_array_equal(context["y_test"], test[1])


def test_direct_eval_runner_returns_nan_predictions_on_failure(split_data):
    train, val, test = split_data
    runner = DummyDirectRunner(train, val, test)

    context = runner.run("raise")

    assert np.isnan(context["y_pred"]).all()
    assert len(context["y_pred"]) == len(val[0])


# TrainEvalRunner


def test_train_eval_runner_run_uses_train_for_fit_and_val_for_eval(split_data):
    train, val, test = split_data
    runner = DummyTrainEvalRunner(train, val, test)

    context = runner.run("model_1")

    assert context["phenotype"] == "model_1"
    np.testing.assert_array_equal(context["y_true"], val[1])
    np.testing.assert_array_equal(context["y_pred"], np.sum(val[0], axis=1))


def test_train_eval_runner_respects_active_eval_split(split_data):
    train, val, test = split_data
    runner = DummyTrainEvalRunner(train, val, test)
    runner.set_eval_split("test")

    context = runner.run("model_1")

    np.testing.assert_array_equal(context["y_true"], test[1])
    np.testing.assert_array_equal(context["y_pred"], np.sum(test[0], axis=1))


def test_train_eval_runner_adds_model_to_context_when_requested(split_data):
    train, val, test = split_data
    runner = DummyTrainEvalRunner(train, val, test)

    context = runner.run("model_1", context_hints={"model"})

    assert "model" in context
    assert isinstance(context["model"], DummyModel)


def test_train_eval_runner_adds_probability_and_score_outputs_when_requested(
    split_data,
):
    train, val, test = split_data
    runner = DummyTrainEvalRunner(train, val, test)

    context = runner.run("model_1", context_hints={"y_pred_proba", "y_pred_score"})

    assert "y_pred_proba" in context
    assert "y_pred_score" in context
    assert context["y_pred_proba"].shape == (len(val[0]), 2)
    assert context["y_pred_score"].shape == (len(val[0]),)


def test_train_eval_runner_adds_feature_importance_when_requested(split_data):
    train, val, test = split_data
    runner = DummyTrainEvalRunner(train, val, test)

    context = runner.run("model_1", context_hints={"feature_importance"})

    assert "feature_importance" in context
    np.testing.assert_array_equal(context["feature_importance"], np.array([0.7, 0.3]))


def test_train_eval_runner_adds_split_specific_keys_when_requested(split_data):
    train, val, test = split_data
    runner = DummyTrainEvalRunner(train, val, test)

    context = runner.run(
        "model_1",
        context_hints={"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"},
    )

    np.testing.assert_array_equal(context["X_train"], train[0])
    np.testing.assert_array_equal(context["y_train"], train[1])
    np.testing.assert_array_equal(context["X_val"], val[0])
    np.testing.assert_array_equal(context["y_val"], val[1])
    np.testing.assert_array_equal(context["X_test"], test[0])
    np.testing.assert_array_equal(context["y_test"], test[1])


# Concrete runner tests


def test_symbolic_regression_runner_run_preserves_common_runner_contract(split_data):
    train, val, test = split_data

    runner = SymbolicRegressionRunner(train, val, test)
    context = runner.run("x0 + x1")

    assert set(context.keys()) == {"phenotype", "y_true", "y_pred"}
    assert context["phenotype"] == "x0 + x1"
    np.testing.assert_array_equal(context["y_true"], val[1])
    np.testing.assert_array_equal(context["y_pred"], np.sum(val[0], axis=1))


def test_symbolic_regression_runner_returns_nan_on_expression_failure(split_data):
    train, val, test = split_data
    runner = SymbolicRegressionRunner(train, val, test)
    context = runner.run("xxx+ddfdf+343")  # incorrect expression
    assert np.isnan(context["y_pred"]).all()


def test_ml_model_runner_run_preserves_common_runner_contract_non_torch(split_data):
    train, val, test = split_data
    parser = DummyModelParser()

    runner = MLModelRunner(
        data_train=train,
        data_val=val,
        data_test=test,
        model_parser=parser,
    )

    context = runner.run("regular_model")

    assert set(context.keys()) == {"phenotype", "y_pred", "y_true"}
    np.testing.assert_array_equal(context["y_true"], val[1])
    np.testing.assert_array_equal(context["y_pred"], np.sum(val[0], axis=1))


def test_ml_model_runner_supports_predict_proba_and_decision_function(split_data):
    train, val, test = split_data
    parser = DummyModelParser()

    runner = MLModelRunner(
        data_train=train,
        data_val=val,
        data_test=test,
        model_parser=parser,
    )

    context = runner.run(
        "regular_model",
        context_hints={"y_pred_proba", "y_pred_score", "model"},
    )

    assert "model" in context
    assert "y_pred_proba" in context
    assert "y_pred_score" in context
    assert context["y_pred_proba"].shape == (len(val[0]), 2)
    assert context["y_pred_score"].shape == (len(val[0]),)


def test_ml_model_runner_torch_path_uses_dataset_based_training_and_prediction(
    dataset_splits,
):
    train, val, test = dataset_splits
    parser = DummyModelParser()

    runner = MLModelRunner(
        data_train=train,
        data_val=val,
        data_test=test,
        model_parser=parser,
    )

    # Force torch-style parser output
    context = runner.run("torch_model", context_hints={"model"})

    assert "model" in context
    assert context["model"].fit_called is True
    assert context["model"].train_dataset is train
    assert context["model"].val_dataset is val
    np.testing.assert_array_equal(context["y_true"], np.array([30.0, 40.0]))
    np.testing.assert_array_equal(context["y_pred"], np.array([11.0, 15.0]))


def test_runner_provided_context_keys_cover_expected_core_keys(split_data):
    train, val, test = split_data

    direct_runner = DummyDirectRunner(train, val, test)
    train_eval_runner = DummyTrainEvalRunner(train, val, test)

    assert {"phenotype", "y_true", "y_pred"}.issubset(
        direct_runner.provided_context_keys
    )
    assert {"phenotype", "y_true", "y_pred", "model"}.issubset(
        train_eval_runner.provided_context_keys
    )
