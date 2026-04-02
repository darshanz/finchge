from finchge.benchmarks.regression.nguyen import Nguyen6Benchmark
from finchge.config import FinchConfig, Keys
from finchge.core import GrammaticalEvolution
from finchge.fitness import FitnessEvaluator, RMSEFitness
from finchge.grammar import GenotypeMapper
from finchge.runners.sr import SymbolicRegressionRunner
from finchge.utils.logger import ExperimentLogger

"""
NOTE: This is a basic example for Nguyen6 regression problem.
Edit the main function as needed to customize and run your experiment.
"""


def main() -> None:
    # setup configs
    config = FinchConfig.from_yaml("config.yaml")

    # get benchmark instance
    benchmark = Nguyen6Benchmark(
        random_state=config.experiment[Keys.RANDOM_SEED],
        train_samples=20,
        test_samples=1000,
    )

    X_train, y_train, X_test, y_test = benchmark.load_data()

    # create a SymbolicRegressionRunner. Runners in FinchGE are used to make prediction for evaluation.
    runner = SymbolicRegressionRunner(
        data_train=(X_train, y_train), data_test=(X_test, y_test)
    )

    # Prepare fitness function. This example uses RMSEFitness fitness.
    # To customize the fitness function, you can create your own class
    # that inherits from finchge.fitness.fitness_functions.GEFitnessFunction
    # and implement the evaluate method.
    fitness_fn = RMSEFitness()
    grammar = benchmark.grammar()

    # Initialize mapper for fitness evaluator
    mapper = GenotypeMapper(
        grammar=grammar,
        max_wraps=config.ge[Keys.MAX_WRAPS],
        max_recursion_depth=config.ge[Keys.MAX_RECURSION_DEPTH],
        random_state=config.experiment[Keys.RANDOM_SEED],
    )

    # Initialize Fitness Evaluator
    fitness_evaluator = FitnessEvaluator(
        fitness_functions=fitness_fn, runner=runner, mapper=mapper
    )

    # setup Experiment Logger
    expt_logger = ExperimentLogger()

    # create Grammatical Evolution instance and run
    ge = GrammaticalEvolution(
        config=config,
        grammar=grammar,
        fitness_evaluator=fitness_evaluator,
        expt_logger=expt_logger,
    )

    result = ge.run()

    print("Best Solution:", result.best_individual.phenotype)
    print("Fitness:", result.best_individual.fitness)


if __name__ == "__main__":
    main()
