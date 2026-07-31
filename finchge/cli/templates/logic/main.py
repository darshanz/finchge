from finchge.benchmarks.logic import MultiplexerBenchmark
from finchge.config import FinchConfig, Keys
from finchge.core import GrammaticalEvolution
from finchge.fitness import FitnessEvaluator
from finchge.fitness.fitness_functions import AccuracyFitness
from finchge.grammar.mapper import GenotypeMapper

"""
NOTE: This is a basic example for Multiplexer-6 logic problem.
Edit the main function as needed to customize and run your experiment.
"""


if __name__ == "__main__":
    # Create benchmark
    benchmark = MultiplexerBenchmark(version=6)

    # Create benchmark runner
    multiplexer_runner = benchmark.create_runner("train")

    # Create fitness function
    fitness = AccuracyFitness()

    # This example uses the default grammar from the benchmark.
    # Custom grammar can be used sing Grammar class eg. Grammar.from_file("grammar.bnf")
    grammar = benchmark.grammar()

    # get project configuration
    ge_config = FinchConfig.from_yaml("config.yaml")

    # create a genotype mapper
    mapper = GenotypeMapper(
        grammar=grammar,
        max_wraps=ge_config.ge[Keys.MAX_WRAPS],
        max_recursion_depth=ge_config.ge[Keys.MAX_RECURSION_DEPTH],
        random_state=ge_config.experiment[Keys.RANDOM_SEED],
    )

    # instantiate fitness evaluator
    fitness_evaluator = FitnessEvaluator(
        runner=multiplexer_runner,
        fitness_functions=fitness,
        mapper=mapper,
        parallel_config=ge_config.parallel,
    )

    # setup run the experiment
    ge_ = GrammaticalEvolution(grammar=grammar, fitness_evaluator=fitness_evaluator)

    result = ge_.run()

    print(f"Best accuracy: {result.all_time_best.fitness[0] * 100:.1f}%")
