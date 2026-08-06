import random

import pytest

from finchge.algorithm import GeneticAlgorithm
from finchge.config import FinchConfig, Keys
from finchge.core.population import Population
from finchge.fitness import FitnessEvaluator
from finchge.fitness.fitness_functions import StringMatchFitness
from finchge.grammar import GenotypeMapper, Grammar
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation import PIGrowInitialiser
from finchge.operators.crossover import SubtreeCrossover
from finchge.operators.mutation import SubtreeMutation
from finchge.operators.replacement import GenerationalReplacement
from finchge.operators.selection import TournamentSelection


@pytest.fixture
def ge_config():
    cfg_dict = {
        "experiment": {
            "random_seed": 42,
            "num_generations": 200,
            "verbose": False,
            "exclude_log": ["genotypes"],
            "cache_type": "lru",
            "cache_size": 128,
        },
        "ge": {
            "grammar_file": "grammar.bnf",
            "population_size": 100,
            "codon_size": 127,
            "max_wraps": 8,
            "max_recursion_depth": 20,
            "genome_length": 100,
            "init_type": "pi-grow",
            "init_max_depth": 20,
            "init_min_depth": 8,
            "max_tree_depth": 20,
            "mutation_probability": 0.01,
            "crossover_probability": 0.75,
            "elite_size": 3,
        },
    }
    return FinchConfig.from_dict(cfg_dict)


@pytest.fixture
def grammar_bnf(ge_config):
    grammar_str = """
            <string> ::= <letter> | <letter> <string>
            <letter> ::= _ | [a-z]

            """

    return Grammar(grammar_str)


def create_deterministic_components(ge_config, grammar_bnf, seed=42):
    """Create all components that share RNG for determinism"""
    shared_rng = random.Random(seed)

    mapper = GenotypeMapper(
        grammar=grammar_bnf,
        max_wraps=ge_config.ge[Keys.MAX_WRAPS],
        max_recursion_depth=ge_config.ge[Keys.MAX_RECURSION_DEPTH],
        random_state=shared_rng,
    )

    evaluator = FitnessEvaluator(
        fitness_functions=StringMatchFitness("hello"),
        mapper=mapper,
    )

    tree_generator = TreeGenerator(
        grammar=grammar_bnf, max_tree_depth=ge_config.ge[Keys.MAX_TREE_DEPTH]
    )
    initialiser = PIGrowInitialiser(
        init_max_depth=12, population_size=10, random_state=shared_rng
    )
    initialiser.set_tree_generator(tree_generator)
    subtree_crossover_ = SubtreeCrossover(
        crossover_proba=ge_config.ge[Keys.CROSSOVER_PROBABILITY],
        non_terminals=grammar_bnf.non_terminals,
    )
    subtree_mutation_ = SubtreeMutation(
        mutation_probability=ge_config.ge[Keys.MUTATION_PROBABILITY],
        tree_generator=tree_generator,
        non_terminals=grammar_bnf.non_terminals,
    )

    algorithm = GeneticAlgorithm(
        selection=TournamentSelection(max_best=False),
        crossover=subtree_crossover_,
        mutation=subtree_mutation_,
        replacement=GenerationalReplacement(max_best=False),
        elite_size=ge_config.ge["elite_size"],
        fitness_evaluator=evaluator,
        random_state=shared_rng,
    )

    return {
        "rng": shared_rng,
        "initialiser": initialiser,
        "mapper": mapper,
        "algorithm": algorithm,
        "evaluator": evaluator,
    }


def test_population_consistency(ge_config, grammar_bnf):
    # For tree based initialisation

    components1 = create_deterministic_components(ge_config, grammar_bnf, seed=42)
    components2 = create_deterministic_components(ge_config, grammar_bnf, seed=42)

    #  Run 1 same seed
    initialiser1 = components1["initialiser"]

    population1 = Population(
        initialiser=initialiser1,
        population_size=ge_config.ge[Keys.POPULATION_SIZE],
    )

    #  run 2 same seed
    initialiser2 = components2["initialiser"]

    population2 = Population(
        initialiser=initialiser2,
        population_size=ge_config.ge[Keys.POPULATION_SIZE],
    )

    # Compare
    assert len(population1) == len(population2)

    # Compare first few individuals
    for i in range(min(3, len(population1))):
        assert (
            population1.individuals[i].genotype is None
        )  # Genotype is none in tree based initialisation
        assert population1.individuals[i].tree == population2.individuals[i].tree


def test_population_consistency_after_evolving(ge_config, grammar_bnf):
    # Test by running two identical utils
    # and see if the population is same after one generation.
    components1 = create_deterministic_components(ge_config, grammar_bnf, seed=42)
    components2 = create_deterministic_components(ge_config, grammar_bnf, seed=42)

    #  Run 1 same seed
    initialiser1 = components1["initialiser"]

    population1 = Population(
        initialiser=initialiser1,
        population_size=ge_config.ge[Keys.POPULATION_SIZE],
    )

    algorithm1 = components1["algorithm"]
    evaluator1 = components1["evaluator"]
    # Evaluate first
    evaluator1.evaluate_population(population1)
    algorithm1.sort_population(population1)

    #  run 2 same seed
    initialiser2 = components2["initialiser"]

    population2 = Population(
        initialiser=initialiser2,
        population_size=ge_config.ge[Keys.POPULATION_SIZE],
    )
    algorithm2 = components2["algorithm"]
    evaluator2 = components2["evaluator"]

    # Evaluate first
    evaluator2.evaluate_population(population2)
    algorithm2.sort_population(population2)

    # COMPARE

    # Compare after first evaluation prior to evolution
    # Compare Population genotype and phenotype
    assert len(population1) == len(population2)

    # Compare first few individuals
    for i in range(min(3, len(population1))):
        assert (
            population1.individuals[i].phenotype == population2.individuals[i].phenotype
        )
        assert population1.individuals[i].fitness == population2.individuals[i].fitness

    for generation in range(10):
        # Evolve One generation
        population1 = algorithm1.evolve_one_generation(population1)
        fittest1 = algorithm1.get_best_individual(population1)

        # Evolve One generation
        population2 = algorithm2.evolve_one_generation(population2)
        fittest2 = algorithm2.get_best_individual(population1)

        # compare fittest phenotypes
        assert fittest1.phenotype == fittest2.phenotype

        # Compare After one generation
        assert len(population1) == len(population2)

        # Compare all individuals
        for i in range(len(population1)):
            assert (
                population1.individuals[i].phenotype
                == population2.individuals[i].phenotype
            )
