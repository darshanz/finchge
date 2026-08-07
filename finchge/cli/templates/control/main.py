from finchge.benchmarks.control import SantaFeTrailBenchmark
from finchge.config import FinchConfig, Keys
from finchge.core import GrammaticalEvolution
from finchge.fitness import FitnessEvaluator
from finchge.fitness.fitness_functions import RewardFitness
from finchge.grammar.mapper import GenotypeMapper

if __name__ == "__main__":
    # instantiate the benchmark
    benchmark = SantaFeTrailBenchmark()
    # prepare benchmark runner
    runner = benchmark.create_runner("train")

    # setup reward fitness function
    fitness = RewardFitness(maximize=True, optimal_fitness=89)

    # This example uses the default grammar from the benchmark.
    # Custom grammar can be used sing Grammar class eg. Grammar.from_file("grammar.bnf")
    grammar = benchmark.grammar()

    # get experiment config
    ge_config = FinchConfig.from_yaml("config.yaml")

    # prepare genotype mapper
    mapper = GenotypeMapper(
        grammar=grammar,
        max_wraps=ge_config.ge[Keys.MAX_WRAPS],
        max_recursion_depth=ge_config.ge[Keys.MAX_RECURSION_DEPTH],
        random_state=ge_config.experiment[Keys.RANDOM_SEED],
    )

    # instantiate fitness evaluator
    fitness_evaluator = FitnessEvaluator(
        runner=runner,
        fitness_functions=fitness,
        mapper=mapper,
        parallel_config=ge_config.parallel,
    )

    #  run experiment
    ge_ = GrammaticalEvolution(grammar=grammar, fitness_evaluator=fitness_evaluator)
    result = ge_.run()

    print(f"Best fitness: {result.all_time_best.fitness[0]}/89 food eaten")
