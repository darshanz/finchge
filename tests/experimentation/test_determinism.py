import random

import pytest

from finchge.algorithm import GeneticAlgorithm
from finchge.config import FinchConfig, Keys
from finchge.core.population import Population
from finchge.fitness import FitnessEvaluator
from finchge.fitness.fitness_functions import StringMatchFitness
from finchge.grammar import GenotypeMapper, Grammar
from finchge.initialisation import RandomGenomeInitialiser
from finchge.operators.crossover import OnePointCrossover
from finchge.operators.mutation import IntFlipMutation
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
            "population_size": 100,
            "grammar_file": "grammar.bnf",
            "codon_size": 127,
            "max_wraps": 8,
            "max_recursion_depth": 20,
            "genome_length": 100,
            "init_type": "rhh",
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

    initialiser = RandomGenomeInitialiser.from_config(ge_config)
    algorithm = GeneticAlgorithm(
        selection=TournamentSelection(max_best=False),
        crossover=OnePointCrossover(
            codon_size=ge_config.ge[Keys.CODON_SIZE],
            crossover_proba=ge_config.ge[Keys.CROSSOVER_PROBABILITY],
        ),
        mutation=IntFlipMutation(
            ge_config.ge[Keys.MUTATION_PROBABILITY],
            codon_size=ge_config.ge[Keys.CODON_SIZE],
        ),
        replacement=GenerationalReplacement(max_best=False),
        elite_size=ge_config.ge[Keys.ELITE_SIZE],
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


# -------- RANDOM GENOME INIT BASED WORK FLOW ---------------------
def test_population_consistency(ge_config, grammar_bnf):
    components1 = create_deterministic_components(ge_config, grammar_bnf, seed=42)
    components2 = create_deterministic_components(ge_config, grammar_bnf, seed=42)

    #  Run 1 same seed
    mapper1 = components1["mapper"]
    initialiser1 = components1["initialiser"]

    population1 = Population(
        initialiser=initialiser1,
        population_size=ge_config.ge[Keys.POPULATION_SIZE],
    )

    #  run 2 same seed
    mapper2 = components2["mapper"]
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
            population1.individuals[i].genotype == population2.individuals[i].genotype
        )
        assert hash(tuple(population1.individuals[i].genotype)) == hash(
            tuple(population2.individuals[i].genotype)
        )

        # map
        mapresult1 = mapper1.map(population1.individuals[i].genotype)
        mapresult2 = mapper2.map(population2.individuals[i].genotype)
        assert mapresult1.phenotype == mapresult2.phenotype


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
    # Evolve One generation
    offsprings1 = algorithm1.evolve_one_generation(population1)

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

    # Evolve One generation
    offsprings2 = algorithm2.evolve_one_generation(population2)

    # COMAPRE
    # Compare after first evaluation prior to evolution

    # Compare Population genotype and phenotype
    assert len(population1) == len(population2)

    # Compare first few individuals
    for i in range(min(3, len(population1))):
        assert (
            population1.individuals[i].genotype == population2.individuals[i].genotype
        )
        assert hash(tuple(population1.individuals[i].genotype)) == hash(
            tuple(population2.individuals[i].genotype)
        )
        assert (
            population1.individuals[i].phenotype == population2.individuals[i].phenotype
        )
        assert population1.individuals[i].fitness == population2.individuals[i].fitness

    # Compare After one generation
    assert len(offsprings1) == len(offsprings2)

    # Compare first few individuals
    for i in range(min(3, len(offsprings1))):
        assert (
            offsprings1.individuals[i].genotype == offsprings2.individuals[i].genotype
        )
        assert hash(tuple(offsprings1.individuals[i].genotype)) == hash(
            tuple(offsprings2.individuals[i].genotype)
        )
        assert (
            offsprings1.individuals[i].phenotype == offsprings2.individuals[i].phenotype
        )
