from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from finchge.fitness.fitness_types import Fitness


class GEFitnessFunction(ABC):
    """
    Base class for a fitness function used in genetic algorithms.

    This abstract class defines the interface for evaluating the fitness
    of a given phenotype. Subclasses must implement the `evaluate` method.

    Args:
        maximize (bool): Indicates whether the goal is to maximize (True)
                         or minimize (False) the fitness score.
    """

    def __init__(
        self,
        maximize: bool = False,
        case_data_key: str | None = None,  # for lexicase support
    ):
        self.maximize = maximize
        self.default_fitness = np.nan  # Use if evaluation fails or is not computable
        self.case_data_key = (
            case_data_key  # case keys should be set to support Lexicase selection
        )

    @property
    def required_context_keys(self) -> set[str]:
        """
        Declare what context keys this fitness function needs.
        Override this in subclasses that need more than y_pred/y_true.
        """
        return {"y_pred", "y_true"}

    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> Fitness:
        """
        Evaluate the fitness score using the provided context.

        Args:
            context (EvaluationContext): EvaluationContext Typeddict containing evaluation inputs,
                            such as 'y_pred', 'y_val', 'x_val', etc.

        Returns:
            Fitness: The computed fitness.
        """
        pass


# Classification


class AccuracyFitness(GEFitnessFunction):
    """
    Fitness function that evaluates model accuracy on a validation set.

    This metric computes the classification accuracy between the predicted labels
    and the ground-truth labels from the validation data. It is intended to be
    used in supervised classification tasks, and is a maximization objective.

    Methods:
        evaluate(context): Computes accuracy using 'y_pred' and 'y_true' from the context.
    """

    def __init__(self) -> None:
        super().__init__(
            maximize=True,
            case_data_key="errors",
        )

    @property
    def required_context_keys(self) -> set[str]:
        return {"y_true", "y_pred"}

    def evaluate(self, context: dict[str, Any]) -> Fitness:
        """
        Evaluates the accuracy of the model's predictions.

        Args:
            context (EvaluationContext): EvaluationContext Typeddict containing evaluation inputs. Must include:
                - 'y_pred': Predicted labels from the model (numpy array or array-like).
                - 'y_test': True labels for the test set (numpy array or array-like).

        Returns:
            float: Accuracy score between 0.0 and 1.0

        Raises:
            ValueError: If y_test and y_pred have different shapes or are empty
        """
        y_true = np.asarray(context["y_true"])
        y_pred = np.asarray(context["y_pred"])

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )

        if y_true.size == 0:
            raise ValueError("Cannot compute accuracy on empty arrays")

        require_case_data = context.get("require_case_data", False)

        # Convert to boolean correctness
        correct = y_true == y_pred

        accuracy = float(np.mean(correct))

        if require_case_data:
            # error = 1 if wrong, 0 if correct
            errors = (~correct).astype(float)
            return Fitness(
                value=accuracy,
                case_data={"errors": errors.tolist()},
            )

        return Fitness(value=accuracy)


class CrossEntropyFitness(GEFitnessFunction):
    """
    CrossEntropy (log loss) fitness for classification.

    Lower values are better (minimize).
    Requires predicted probabilities.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__(
            maximize=False,
            case_data_key="errors",
        )
        self.eps = eps

    @property
    def required_context_keys(self) -> set[str]:
        return {"y_true", "y_pred_proba"}

    def evaluate(self, context: dict[str, Any]) -> Fitness:
        y_true = np.asarray(context["y_true"])
        y_proba = np.asarray(context["y_pred_proba"])

        if y_true.shape[0] != y_proba.shape[0]:
            raise ValueError(
                f"Mismatch: y_true len {len(y_true)} vs y_proba {y_proba.shape}"
            )

        if y_true.size == 0:
            raise ValueError("Cannot compute log loss on empty arrays")

        require_case_data = context.get("require_case_data", False)

        # Clip probabilities for numerical stability
        y_proba = np.clip(y_proba, self.eps, 1 - self.eps)

        # Binary classification
        if y_proba.ndim == 1 or y_proba.shape[1] == 1:
            p = y_proba.reshape(-1)
            errors = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

        else:
            # Multi-class case
            # y_true assumed to be class indices
            probs = y_proba[np.arange(len(y_true)), y_true]
            errors = -np.log(probs)

        loss_ = float(np.mean(errors))

        if require_case_data:
            return Fitness(
                value=loss_,
                case_data={"errors": errors.tolist()},
            )

        return Fitness(value=loss_)


class HingeLossFitness(GEFitnessFunction):
    """
    Mean Hinge Loss fitness for binary classification problems. This fitness evaluates how well a model separates
    two classes by measuring the margin between predictions and true labels. Hinge loss is calculated as
    hinge_i = max(0, 1 - y_true_i * y_pred_score_i)  Should be used only for margin-based classifiers.
    The ground truth should be encoded between -1 and 1 ,
    not 0/1. Similarly, the prediction scores should be raw model outputs (real-valued scores),
    representing the model’s confidence or margin, not class labels.
    The overall fitness value is the mean hinge loss across all samples.
    """

    def __init__(self) -> None:
        super().__init__(
            maximize=False,
            case_data_key="errors",
        )

    @property
    def required_context_keys(self) -> set[str]:
        return {"y_true", "y_pred_score"}

    def evaluate(self, context: dict[str, Any]) -> Fitness:
        """

        Args:
            context: Context containing `y_true` labels encoded as -1 or +1,
                and `y_pred_score` with raw model outputs.

        Returns:
            Fitness value for the hinge loss.

        """
        y_true = np.asarray(context["y_true"])
        y_score = np.asarray(context["y_pred_score"])

        if y_true.shape != y_score.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred_score {y_score.shape}"
            )

        if y_true.size == 0:
            raise ValueError("Cannot compute hinge loss on empty arrays")

        unique_labels = set(np.unique(y_true).tolist())
        if not unique_labels.issubset({-1, 1}):
            raise ValueError("HingeLossFitness requires y_true labels in {-1, +1}")

        require_case_data = context.get("require_case_data", False)

        if np.isnan(y_score).any():
            infs = [float("inf")] * len(y_true)
            if require_case_data:
                return Fitness(
                    value=float("inf"),
                    case_data={"errors": infs},
                )
            return Fitness(value=float("inf"))

        errors = np.maximum(0.0, 1.0 - y_true * y_score)
        mean_hinge = float(np.mean(errors))

        if require_case_data:
            return Fitness(
                value=mean_hinge,
                case_data={"errors": errors.tolist()},
            )

        return Fitness(value=mean_hinge)


# Regression
class MAEFitness(GEFitnessFunction):
    """
    Mean Absolute Error fitness for regression problems.
    Lower values indicate better fit (minimization objective).
    """

    def __init__(self) -> None:
        super().__init__(
            maximize=False,
            case_data_key="errors",
        )

    @property
    def required_context_keys(self) -> set[str]:
        return {"y_true", "y_pred"}

    def evaluate(self, context: dict[str, Any]) -> Fitness:
        """
        Calculate Mean Absolute Error between true and predicted values.

        Args:
            context: Must contain:
                - 'y_true': Ground truth values (array-like)
                - 'y_pred': Predicted values (array-like)

        """
        y_true = np.asarray(context["y_true"])
        y_pred = np.asarray(context["y_pred"])

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )

        if y_true.size == 0:
            raise ValueError("Cannot compute MAE on empty arrays")

        if np.isnan(y_pred).any():
            if context.get("require_case_data", False):
                return Fitness(
                    value=float("inf"),
                    case_data={"errors": [float("inf")] * len(y_true)},
                )
            return Fitness(value=float("inf"))

        abs_errors = np.abs(y_true - y_pred)
        mae = float(np.mean(abs_errors))

        if context.get("require_case_data", False):
            return Fitness(
                value=mae,
                case_data={"errors": abs_errors.tolist()},
            )

        return Fitness(value=mae)


class MSEFitness(GEFitnessFunction):
    def __init__(self) -> None:
        super().__init__(maximize=False, case_data_key="errors")

    @property
    def required_context_keys(self) -> set[str]:
        return {"y_true", "y_pred"}

    def evaluate(self, context: dict[str, Any]) -> Fitness:
        y_true = np.asarray(context["y_true"])
        y_pred = np.asarray(context["y_pred"])

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        if y_true.size == 0:
            raise ValueError("Cannot compute MSE on empty arrays")

        require_case_data = context.get("require_case_data", False)

        if np.isnan(y_pred).any():
            if require_case_data:
                return Fitness(
                    value=float("inf"),
                    case_data={"errors": [float("inf")] * len(y_true)},
                )
            return Fitness(value=float("inf"))

        squared_errors = (y_true - y_pred) ** 2
        mse = float(np.mean(squared_errors))

        if require_case_data:
            return Fitness(
                value=mse,
                case_data={"errors": squared_errors.tolist()},
            )

        return Fitness(value=mse)


class RMSEFitness(GEFitnessFunction):
    def __init__(self) -> None:
        super().__init__(
            maximize=False,
            case_data_key="errors",
        )

    def evaluate(self, context: dict[str, Any]) -> Fitness:
        y_true = context["y_true"]
        y_pred = context["y_pred"]
        if np.isnan(y_pred).any():
            # Check for NaN set very high RMSE for discarded ones.
            return Fitness(value=np.inf)

        # To fix RuntimeWarning: overflow encountered in square
        # Clip predictions to keep RMSE stable.
        y_pred = np.clip(y_pred, -1e10, 1e10)
        residuals = y_true - y_pred
        squared_errors = residuals**2
        rmse: np.float64 = np.sqrt(np.mean(squared_errors))

        # When Lexicase Selection is used
        if context.get("require_case_data", False):
            return Fitness(
                value=float(rmse),
                case_data={"errors": np.abs(residuals).tolist()},
            )

        return Fitness(value=float(rmse))


class RewardFitness(GEFitnessFunction):
    """
    Fitness function for control problems with reward maximization.
    Simply returns the reward (higher is better).
    """

    def __init__(self, maximize: bool = True, optimal_fitness: Optional[float] = None):
        """
        Initialize reward fitness.

        Args:
            maximize: Whether to maximize (True) or minimize (False)
            optimal_fitness: Known optimal fitness value
        """
        super().__init__(maximize=maximize)
        self.optimal_fitness = optimal_fitness
        self.name = "RewardFitness"

    @property
    def required_context_keys(self) -> set[str]:
        # only needs y_pred
        return {"y_pred"}

    def evaluate(self, context: dict[str, Any]) -> Fitness:
        """
        Evaluate fitness from context.

        Expected context keys:
            y_pred: Array of rewards from runner

        Args:
            context: context with y_pred

        Returns:
            Total reward (sum across episodes)
        """
        y_pred = context.get("y_pred", np.array([0.0]))

        # Sum rewards across episodes
        total_reward = float(np.sum(y_pred))

        return Fitness(value=total_reward)

    def __repr__(self) -> str:
        direction = "maximize" if self.maximize else "minimize"
        return f"RewardFitness({direction})"


class StringMatchFitness(GEFitnessFunction):
    def __init__(self, target: str):
        super().__init__(maximize=False)
        self.target = target
        self.target_len: int = len(target)

    def evaluate(self, context: dict[str, Any]) -> Fitness:
        phenotype = context["phenotype"]

        max_len = max(self.target_len, len(phenotype))
        min_len = min(self.target_len, len(phenotype))

        matches: int = sum(
            t == g for t, g in zip(self.target[:min_len], phenotype[:min_len])
        )

        return Fitness(value=(max_len - matches))
