from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class GEFitnessFunction(ABC):
    """
    Base class for a fitness function used in genetic algorithms.

    This abstract class defines the interface for evaluating the fitness
    of a given phenotype. Subclasses must implement the `evaluate` method.

    Args:
        maximize (bool): Indicates whether the goal is to maximize (True)
                         or minimize (False) the fitness score.
    """

    def __init__(self, maximize: bool = False):
        self.maximize = maximize
        self.default_fitness = np.nan  # Use if evaluation fails or is not computable

    @property
    def required_context_keys(self) -> set[str]:
        """
        Declare what context keys this fitness function needs.
        Override this in subclasses that need more than y_pred/y_true.
        """
        return {"y_pred", "y_true"}

    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> float:
        """
        Evaluate the fitness score using the provided context.

        Args:
            context (EvaluationContext): EvaluationContext Typeddict containing evaluation inputs,
                            such as 'y_pred', 'y_val', 'x_val', etc.

        Returns:
            float: The computed fitness score.
        """
        pass


class AccuracyFitness(GEFitnessFunction):
    """
    Fitness function that evaluates model accuracy on a validation set.

    This metric computes the classification accuracy between the predicted labels
    and the ground-truth labels from the validation data. It is intended to be
    used in supervised classification tasks, and is a maximization objective.

    Inherits from:
        GEFitnessFunction (maximize=True)

    Methods:
        evaluate(context): Computes accuracy using 'y_pred' and 'y_val' from the context.
    """

    def __init__(self) -> None:
        super().__init__(maximize=True)

    def evaluate(self, context: dict[str, Any]) -> float:
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
        y_test = context["y_true"]
        y_pred = context["y_pred"]

        # Ensure we have numpy arrays
        if not isinstance(y_test, np.ndarray):
            y_test_arr = np.array(y_test)
        else:
            y_test_arr = y_test

        if not isinstance(y_pred, np.ndarray):
            y_pred_arr = np.array(y_pred)
        else:
            y_pred_arr = y_pred

        # Check shapes match
        if y_test_arr.shape != y_pred_arr.shape:
            raise ValueError(
                f"Shapes of y_test ({y_test_arr.shape}) and y_pred ({y_pred_arr.shape}) "
                f"do not match"
            )

        # Check for empty arrays
        if y_test_arr.size == 0:
            raise ValueError("Input arrays cannot be empty")

        # Calculate accuracy
        correct_predictions = np.sum(y_test_arr == y_pred_arr)
        total_predictions = y_test_arr.size

        return float(correct_predictions / total_predictions)


class MAEFitness(GEFitnessFunction):
    """
    Mean Absolute Error fitness for regression problems.
    Lower values indicate better fit (minimization objective).
    """

    def __init__(self) -> None:
        super().__init__(maximize=False)

    @property
    def required_context_keys(self) -> set[str]:
        return {"y_true", "y_pred"}

    def evaluate(self, context: dict[str, Any]) -> float:
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
            return np.inf

        mae = np.mean(np.abs(y_true - y_pred))
        return float(mae)


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

    def evaluate(self, context: dict[str, Any]) -> float:
        """
        Evaluate fitness from context.

        Expected context keys:
            - y_pred: Array of rewards from runner
            - y_true: Target values (usually zeros)

        Args:
            context: Dictionary with evaluation results

        Returns:
            Total reward (sum across episodes)
        """
        y_pred = context.get("y_pred", np.array([0.0]))

        # Sum rewards across episodes
        total_reward = float(np.sum(y_pred))

        return total_reward

    def __repr__(self) -> str:
        direction = "maximize" if self.maximize else "minimize"
        return f"RewardFitness({direction})"


class RMSEFitness(GEFitnessFunction):
    def __init__(self) -> None:
        super().__init__(maximize=False)

    def evaluate(self, context: dict[str, Any]) -> float:
        y_true = context["y_true"]
        y_pred = context["y_pred"]
        if np.isnan(y_pred).any():
            # Check for NaN set very high RMSE for discarded ones.
            return np.inf
        rmse: np.float64 = np.sqrt(np.mean((y_true - y_pred) ** 2))
        fitness = np.inf if np.isnan(rmse) else rmse  # Return Inf if rmse is NaN
        return float(fitness)


class StringMatchFitness(GEFitnessFunction):
    def __init__(self, target: str):
        super().__init__(maximize=False)
        self.target = target
        self.target_len = len(target)

    def evaluate(self, context: dict[str, Any]) -> int:
        phenotype = context["phenotype"]

        max_len = max(self.target_len, len(phenotype))
        min_len = min(self.target_len, len(phenotype))

        matches = sum(
            t == g for t, g in zip(self.target[:min_len], phenotype[:min_len])
        )

        return max_len - matches
