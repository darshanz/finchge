from finchge.algorithm.nsga import NSGA2
from finchge.config import FinchConfig
from finchge.core.engine import GrammaticalEvolution
from finchge.fitness import FitnessEvaluator, GEFitnessFunction
from finchge.fitness.fitness_types import Fitness
from finchge.grammar import GenotypeMapper, Grammar
from finchge.operators.crossover import OnePointCrossover
from finchge.operators.mutation import IntFlipMutation
from finchge.operators.replacement import NSGA2ElitistReplacement
from finchge.operators.selection import NSGA2TournamentSelection


class PhenotypeLengthFitness(GEFitnessFunction):
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


def test_engine_multi_objective_run_does_not_request_single_best_individual():
    grammar = Grammar(
        """
        <expr> ::= a | bb | ccc
        """
    )

    config = FinchConfig.from_dict(
        {
            "experiment": {
                "random_seed": 42,
                "num_generations": 1,
                "verbose": False,
            },
            "ge": {
                "population_size": 6,
                "codon_size": 127,
                "genome_length": 4,
                "max_wraps": 0,
                "max_recursion_depth": 10,
                "max_tree_depth": 10,
                "init_type": "random_genome",
                "mutation_probability": 0.1,
                "crossover_probability": 0.7,
                "elite_size": 0,
            },
        }
    )

    mapper = GenotypeMapper(
        grammar=grammar,
        max_wraps=config.ge["max_wraps"],
        max_recursion_depth=config.ge["max_recursion_depth"],
        random_state=42,
    )

    evaluator = FitnessEvaluator(
        fitness_functions=[
            PhenotypeLengthFitness(),
            ContainsAFitness(),
        ],
        mapper=mapper,
    )

    algorithm = NSGA2(
        selection=NSGA2TournamentSelection(random_state=42),
        crossover=OnePointCrossover(
            codon_size=config.ge["codon_size"],
            crossover_proba=config.ge["crossover_probability"],
            random_state=42,
        ),
        mutation=IntFlipMutation(
            mutation_probability=config.ge["mutation_probability"],
            codon_size=config.ge["codon_size"],
            random_state=42,
        ),
        replacement=NSGA2ElitistReplacement(
            maximize_flags=evaluator.get_maximize_flags(),
            random_state=42,
        ),
        elite_size=config.ge["elite_size"],
        fitness_evaluator=evaluator,
        random_state=42,
    )

    ge = GrammaticalEvolution(
        fitness_evaluator=evaluator,
        grammar=grammar,
        config=config,
        algorithm=algorithm,
        random_state=42,
    )

    result = ge.run()

    assert result.best_individual is None
    assert result.pareto_front is not None
    assert len(result.pareto_front) > 0
