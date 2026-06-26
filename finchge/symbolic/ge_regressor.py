from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

from finchge.config.config import FinchConfig, Keys
from finchge.core.engine import GrammaticalEvolution
from finchge.core.individual import Individual
from finchge.core.result import GEResult
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.fitness.fitness_functions import GEFitnessFunction
from finchge.grammar import GenotypeMapper, Grammar
from finchge.runners.sr import SymbolicRegressionRunner
from finchge.symbolic.expression import SymbolicExpression
from finchge.utils.logger import ExperimentLogger
from finchge.utils.random_mixin import RandomStateMixin


class GERegressor(RandomStateMixin, BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """
    Scikit-learn compatible symbolic regression model based on
    Grammatical Evolution (GE).

    The estimator evolves mathematical expressions defined by a grammar
    and selects the best-performing individual according to the provided
    fitness function(s). The resulting symbolic expression can then be
    evaluated to make predictions.

    Attributes:
        ge_result_ (GEResult): Result object returned by the GE run.
        selected_individual_ (Individual | None): Individual currently
            selected for prediction.
        selected_expression_ (SymbolicExpression | None): Parsed symbolic
            expression corresponding to the selected individual.
    """

    def __init__(
        self,
        grammar: Grammar,
        config: FinchConfig | dict[str, Any],
        fitness_functions: GEFitnessFunction | list[GEFitnessFunction],
        generations: int = 100,
        population_size: int = 100,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the symbolic regression estimator.

        Args:
            grammar (Grammar): Grammar defining the search space of valid
                symbolic expressions.
            config (FinchConfig | dict[str, Any]): Configuration controlling
                GE behavior (operators, mutation rates, limits, etc.).
                A dictionary will be converted into a ``FinchConfig``.
            fitness_functions (GEFitnessFunction | list[GEFitnessFunction]):
                Fitness function(s) used to evaluate candidate expressions.
            generations (int, optional): Number of evolutionary generations.
                Defaults to 100.
            population_size (int, optional): Number of individuals per
                generation. Defaults to 100.
            random_state (int | None, optional): Random seed for reproducibility.
        """

        super().__init__(random_state=random_state)
        self.grammar = grammar
        self.generations = generations
        self.population_size = population_size

        # learned attributes
        self.best_individual_: Optional[Individual] = None
        self.best_phenotype_: Optional[str] = None

        self.random_state = random_state

        self.fitness_functions = fitness_functions

        if isinstance(config, FinchConfig):
            self.config = config
        else:
            self.config = FinchConfig.from_dict(config)

        self.genotype_mapper = GenotypeMapper(
            grammar=self.grammar,
            max_wraps=self.config.ge[Keys.MAX_WRAPS],
            max_recursion_depth=self.config.ge[Keys.MAX_RECURSION_DEPTH],
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any) -> "GERegressor":
        """
        Run grammatical evolution to discover a symbolic expression that
        fits the provided data.

        This launches the evolutionary search, evaluates individuals using
        the configured fitness function(s), and stores the resulting GE run
        along with the best individual (if available).

        Args:
            X (Any): Training features.
            y (Any): Target values.

        Returns:
            GERegressor: The fitted estimator.
        """
        sym_runner = SymbolicRegressionRunner(
            data_train=(X, y),
        )
        fitness_evaluator = FitnessEvaluator(
            runner=sym_runner,
            fitness_functions=self.fitness_functions,
            mapper=self.genotype_mapper,
            parallel_config=self.config.parallel,
        )
        expt_logger = ExperimentLogger()
        ge = GrammaticalEvolution(
            grammar=self.grammar,
            fitness_evaluator=fitness_evaluator,
            config=self.config,
            expt_logger=expt_logger,
            random_state=self.random_state,
        )

        ge_result: GEResult = ge.run()

        self.ge_result_ = ge_result

        # there will be best individual if it is single objective
        self.selected_individual_ = ge_result.best_individual

        if self.selected_individual_ is not None:
            phenotype = (
                self.selected_individual_.phenotype
                if self.selected_individual_.phenotype
                else ""
            )
            self.selected_expression_: SymbolicExpression | None = SymbolicExpression(
                phenotype
            )
        else:
            self.selected_expression_ = None

        return self

    def select_individual(self, individual: Individual) -> None:
        """
        Manually select an individual from the GE population for prediction.

        Useful for multi-objective runs where no single best individual is
        automatically chosen.

        Args:
            individual (Individual): Individual whose phenotype will be used
                as the prediction expression.
        """
        self.selected_individual_ = individual
        if not individual.phenotype:
            raise ValueError("Phenotype can not be None")
        self.selected_expression_ = SymbolicExpression(individual.phenotype)

    def predict(self, X: Any) -> NDArray[np.float64]:
        """
        Evaluate the selected symbolic expression on the input data.

        Args:
            X (Any): Input features.

        Returns:
            NDArray[np.float64]: Predicted target values.

        Raises:
            RuntimeError: If the estimator has not been fitted or no
                individual has been selected.
        """
        if self.selected_expression_ is None:
            raise RuntimeError("Model has not been fitted or no valid model selected")

        return self.selected_expression_.eval(X)
