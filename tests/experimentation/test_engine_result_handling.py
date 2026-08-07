import pytest

from finchge.algorithm.base import BaseAlgorithm
from finchge.config import FinchConfig
from finchge.core.engine import GrammaticalEvolution
from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.fitness import FitnessEvaluator, GEFitnessFunction
from finchge.fitness.fitness_types import Fitness
from finchge.grammar import GenotypeMapper, Grammar


class LengthFitness(GEFitnessFunction):
    def __init__(self):
        super().__init__(maximize=False)

    @property
    def required_context_keys(self):
        return set()

    def evaluate(self, context):
        return Fitness(value=float(len(context["phenotype"])))


class ContainsAFitness(GEFitnessFunction):
    def __init__(self):
        super().__init__(maximize=True)

    @property
    def required_context_keys(self):
        return set()

    def evaluate(self, context):
        return Fitness(value=1.0 if "a" in context["phenotype"] else 0.0)


class DummyEngineAlgorithm(BaseAlgorithm):
    def inject_operator_rng(self):
        pass


class FinalPopulationAlgorithm(DummyEngineAlgorithm):
    def sort_population(self, population):
        pass

    def get_best_individual(self, population):
        return population.individuals[0]

    def get_pareto_front(self, population):
        raise NotImplementedError

    def evolve_one_generation(self, population):
        ind = Individual.from_genotype([0])
        ind.phenotype = "final"
        ind.fitness = [0.0]
        return Population.from_individuals([ind], population_size=1)


class FinalFrontAlgorithm(DummyEngineAlgorithm):
    def sort_population(self, population):
        pass

    def get_best_individual(self, population):
        raise NotImplementedError

    def get_pareto_front(self, population):
        return population.individuals

    def evolve_one_generation(self, population):
        ind = Individual.from_genotype([0])
        ind.phenotype = "final"
        ind.fitness = [0.0, 1.0]
        return Population.from_individuals([ind], population_size=1)


@pytest.fixture
def engine_config():
    return FinchConfig.from_dict(
        {
            "experiment": {
                "random_seed": 42,
                "num_generations": 1,
                "verbose": False,
            },
            "ge": {
                "population_size": 1,
                "codon_size": 127,
                "genome_length": 4,
                "max_wraps": 0,
                "max_recursion_depth": 10,
                "max_tree_depth": 10,
                "init_type": "random_genome",
                "mutation_probability": 0.0,
                "crossover_probability": 0.0,
                "elite_size": 0,
            },
        }
    )


@pytest.fixture
def tiny_grammar():
    return Grammar("<expr> ::= a | bb")


def test_single_objective_result_contains_final_population(engine_config, tiny_grammar):
    mapper = GenotypeMapper(grammar=tiny_grammar, random_state=42)
    evaluator = FitnessEvaluator(
        fitness_functions=LengthFitness(),
        mapper=mapper,
    )

    ge = GrammaticalEvolution(
        fitness_evaluator=evaluator,
        grammar=tiny_grammar,
        config=engine_config,
        algorithm=FinalPopulationAlgorithm(random_state=42),
        random_state=42,
    )

    result = ge.run()

    assert result.best_in_generation.phenotype == "final"
    assert result.population.individuals[0].phenotype == "final"


def test_multi_objective_result_contains_final_pareto_front(
    engine_config, tiny_grammar
):
    mapper = GenotypeMapper(grammar=tiny_grammar, random_state=42)
    evaluator = FitnessEvaluator(
        fitness_functions=[LengthFitness(), ContainsAFitness()],
        mapper=mapper,
    )

    assert evaluator.is_multi_objective()

    ge = GrammaticalEvolution(
        fitness_evaluator=evaluator,
        grammar=tiny_grammar,
        config=engine_config,
        algorithm=FinalFrontAlgorithm(random_state=42),
        random_state=42,
    )

    result = ge.run()

    assert result.best_in_generation is None
    assert result.pareto_front[0].phenotype == "final"
    assert result.population.individuals[0].phenotype == "final"
