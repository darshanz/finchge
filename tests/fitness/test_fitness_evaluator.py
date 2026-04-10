import numpy as np
import pytest

from finchge.algorithm import GeneticAlgorithm
from finchge.core import Individual
from finchge.fitness import FitnessEvaluator, GEFitnessFunction
from finchge.fitness.fitness_types import Fitness
from finchge.grammar import GenotypeMapper, Grammar
from finchge.initialisation import RandomGenomeInitialiser
from finchge.operators.crossover import OnePointCrossover
from finchge.operators.mutation import IntFlipMutation
from finchge.operators.replacement import GenerationalReplacement
from finchge.operators.selection import LexicaseSelection, TournamentSelection
from finchge.runners import PhenotypeRunner

# Some DummyFitness functions to generalize the test cases.
# Testing with dummy runners as they can cover the real runners
# both currently existing and custom ones that can be built later


class NeedsPredFitness(GEFitnessFunction):
    def __init__(self):
        super().__init__(maximize=False)

    @property
    def required_context_keys(self):
        return {"y_true", "y_pred"}

    def evaluate(self, context):
        return Fitness(value=0.0)


class NeedsProbaFitness(GEFitnessFunction):
    def __init__(self):
        super().__init__(maximize=False)

    @property
    def required_context_keys(self):
        return {"y_true", "y_pred_proba"}

    def evaluate(self, context):
        return Fitness(value=0.0)


class CasewiseFitness(GEFitnessFunction):
    def __init__(self):
        super().__init__(maximize=False, case_data_key="errors")

    @property
    def required_context_keys(self):
        return {"y_true", "y_pred"}

    def evaluate(self, context):
        y_true = np.asarray(context["y_true"])
        y_pred = np.asarray(context["y_pred"])
        errors = np.abs(y_true - y_pred)
        if context.get("require_case_data", False):
            return Fitness(
                value=float(np.mean(errors)), case_data={"errors": errors.tolist()}
            )
        return Fitness(value=float(np.mean(errors)))


class NoCasewiseFitness(GEFitnessFunction):
    def __init__(self):
        super().__init__(maximize=False, case_data_key=None)

    @property
    def required_context_keys(self):
        return {"y_true", "y_pred"}

    def evaluate(self, context):
        return Fitness(value=0.0)


class RunnerWithPred(PhenotypeRunner):
    @property
    def provided_context_keys(self):
        return {"y_true", "y_pred"}

    def run(self, phenotype, context_hints=None):
        return {
            "y_true": np.array([1.0, 2.0]),
            "y_pred": np.array([1.1, 1.9]),
        }


class RunnerWithProba(PhenotypeRunner):
    @property
    def provided_context_keys(self):
        return {"y_true", "y_pred", "y_pred_proba"}

    def run(self, phenotype, context_hints=None):
        return {
            "y_true": np.array([0, 1]),
            "y_pred": np.array([0, 1]),
            "y_pred_proba": np.array([[0.8, 0.2], [0.1, 0.9]]),
        }


class LyingRunner(PhenotypeRunner):
    # The runner lies that it provides the context keys
    # but does not provide at run time
    @property
    def provided_context_keys(self):
        return {"y_true", "y_pred"}

    def run(self, phenotype, context_hints=None):
        return {
            "y_true": np.array([1.0, 2.0]),
            # missing y_pred on purpose
        }


@pytest.fixture
def simple_grammar():
    grammar_str = """
    <expr> ::= <expr> <op> <expr> | <var> | <const>
    <op> ::= + | -
    <var> ::= x
    <const> ::= 1.0 | 2.0
    """
    return Grammar(grammar_str)


@pytest.fixture
def mapper(simple_grammar):
    return GenotypeMapper(grammar=simple_grammar, random_state=1234)


def test_fitness_evaluator_accepts_valid_runner_fitness_combination(mapper):
    evaluator = FitnessEvaluator(
        fitness_functions=[NeedsPredFitness()],
        mapper=mapper,
        runner=RunnerWithPred(),
    )
    assert evaluator is not None


def test_fitness_evaluator_rejects_missing_required_context_keys(mapper):
    with pytest.raises(ValueError, match="y_pred_proba"):
        FitnessEvaluator(
            fitness_functions=[NeedsProbaFitness()],
            mapper=mapper,
            runner=RunnerWithPred(),
        )


def test_fitness_evaluator_accepts_runner_with_probability_support(mapper):
    evaluator = FitnessEvaluator(
        fitness_functions=[NeedsProbaFitness()],
        mapper=mapper,
        runner=RunnerWithProba(),
    )
    assert evaluator is not None


def test_fitness_evaluator_accepts_case_data_when_supported(mapper):
    evaluator = FitnessEvaluator(
        fitness_functions=[CasewiseFitness()],
        mapper=mapper,
        runner=RunnerWithPred(),
        require_case_data=True,
    )
    assert evaluator.require_case_data is True


def test_fitness_evaluator_rejects_case_data_requirement_when_fitness_cannot_provide_it(
    mapper,
):
    with pytest.raises(ValueError, match="case data"):
        FitnessEvaluator(
            fitness_functions=[NoCasewiseFitness()],
            mapper=mapper,
            runner=RunnerWithPred(),
            require_case_data=True,
        )


def test_fitness_evaluator_runtime_validation_catches_missing_context_keys(mapper):
    # even if the runner says it provides the context keys but does't provides them during evaluation
    # such runner should be caught at runtime.
    evaluator = FitnessEvaluator(
        fitness_functions=[NeedsPredFitness()],
        mapper=mapper,
        runner=LyingRunner(),
    )
    initialiser = RandomGenomeInitialiser(
        genome_length=100, codon_size=127, random_state=42
    )
    ind = initialiser.initialise()

    with pytest.raises(ValueError, match="y_pred"):
        evaluator.evaluate_individual(ind)


# tests for fitness evaluator to see whether
# the combination of algorithm, fitness_function and selection method is OK


def test_fitness_evaluator_raise_warning_when_evaluator_requires_selection_does_not_need(
    mapper,
):
    # IF fitness evaluator requires the case data, fitness functions may calculate it
    # But if selection strategy does not need case data. This can be allowed but
    # warning should be shown because that may be using unnecessary computation to generate case data
    evaluator = FitnessEvaluator(
        fitness_functions=[CasewiseFitness()],
        mapper=mapper,
        runner=RunnerWithPred(),
        require_case_data=True,
    )
    with pytest.warns(
        UserWarning,
        match="FitnessEvaluator is configured to compute case data, "
        "but the selected selection strategy does not use it.",
    ):
        GeneticAlgorithm(
            selection=TournamentSelection(max_best=False, tournament_size=2),
            fitness_evaluator=evaluator,
            crossover=OnePointCrossover(127, 0.5),
            mutation=IntFlipMutation(0.5, 127),
            replacement=GenerationalReplacement(max_best=True, random_state=1234),
            elite_size=3,
        )


def test_algorithm_rejects_lexicase_when_evaluator_not_configured_for_case_data(mapper):
    evaluator = FitnessEvaluator(
        fitness_functions=[CasewiseFitness()],
        mapper=mapper,
        runner=RunnerWithPred(),
        require_case_data=False,
    )
    with pytest.raises(ValueError, match="require_case_data"):
        GeneticAlgorithm(
            selection=LexicaseSelection(case_key="errors"),
            fitness_evaluator=evaluator,
            crossover=OnePointCrossover(127, 0.5),
            mutation=IntFlipMutation(0.5, 127),
            replacement=GenerationalReplacement(max_best=True, random_state=1234),
            elite_size=3,
        )


def test_algorithm_allows_non_casewise_selection_with_case_data_enabled(mapper):
    evaluator = FitnessEvaluator(
        fitness_functions=[CasewiseFitness()],
        mapper=mapper,
        runner=RunnerWithPred(),
        require_case_data=False,
    )

    GeneticAlgorithm(
        selection=TournamentSelection(max_best=False, tournament_size=2),
        fitness_evaluator=evaluator,
        crossover=OnePointCrossover(127, 0.5),
        mutation=IntFlipMutation(0.5, 127),
        replacement=GenerationalReplacement(max_best=True, random_state=1234),
        elite_size=3,
    )


def test_algorithm_accepts_lexicase_with_casewise_fitness(mapper):
    evaluator = FitnessEvaluator(
        fitness_functions=[CasewiseFitness()],
        mapper=mapper,
        runner=RunnerWithPred(),
        require_case_data=True,
    )

    algorithm = GeneticAlgorithm(
        selection=LexicaseSelection(case_key="errors"),
        fitness_evaluator=evaluator,
        crossover=OnePointCrossover(127, 0.5),
        mutation=IntFlipMutation(0.5, 127),
        replacement=GenerationalReplacement(max_best=True, random_state=1234),
        elite_size=3,
    )

    # intitializes successfully
    assert algorithm is not None


def test_algorithm_rejects_missing_case_data_on_individuals(mapper):
    evaluator = FitnessEvaluator(
        fitness_functions=[CasewiseFitness()],
        mapper=mapper,
        runner=RunnerWithPred(),
        require_case_data=True,
    )

    algorithm = GeneticAlgorithm(
        selection=LexicaseSelection(case_key="errors"),
        fitness_evaluator=evaluator,
        crossover=OnePointCrossover(127, 0.5),
        mutation=IntFlipMutation(0.5, 127),
        replacement=GenerationalReplacement(max_best=True, random_state=1234),
        elite_size=3,
    )

    initialiser = RandomGenomeInitialiser(
        genome_length=100, codon_size=127, random_state=42
    )
    ind = initialiser.initialise()
    ind.invalid = False
    ind.fitness = [0.5]
    # no case metadata set

    with pytest.raises(ValueError, match="casewise"):
        algorithm._validate_selection_requirements([ind])


def test_algorithm_rejects_missing_required_case_key(mapper):
    evaluator = FitnessEvaluator(
        fitness_functions=[CasewiseFitness()],
        mapper=mapper,
        runner=RunnerWithPred(),
        require_case_data=True,
    )

    algorithm = GeneticAlgorithm(
        selection=LexicaseSelection(case_key="errors"),
        fitness_evaluator=evaluator,
        crossover=OnePointCrossover(127, 0.5),
        mutation=IntFlipMutation(0.5, 127),
        replacement=GenerationalReplacement(max_best=True, random_state=1234),
        elite_size=3,
    )

    initialiser = RandomGenomeInitialiser(
        genome_length=100, codon_size=127, random_state=42
    )
    ind = initialiser.initialise()
    ind.invalid = False
    ind.fitness = [0.5]
    ind.set_meta(Individual.CASE_DATA_META_KEY, {"losses": [0.1, 0.2]})

    with pytest.raises(ValueError, match="errors"):
        algorithm._validate_selection_requirements([ind])
