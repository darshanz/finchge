from finchge.config import FinchConfig, Keys
from finchge.core.engine import GrammaticalEvolution
from finchge.fitness import FitnessEvaluator
from finchge.fitness.fitness_functions import StringMatchFitness
from finchge.grammar import Grammar
from finchge.grammar.mapper import GenotypeMapper


def main() -> None:
    # setup configs
    config = FinchConfig.from_yaml("config.yaml")

    # Load Grammar
    grammar = Grammar.from_file("grammar.bnf")

    # Prepare fitness function
    fitness_fn = StringMatchFitness(target="finch")

    # Initialize mapper for fitness evaluator
    mapper = GenotypeMapper(
        grammar=grammar,
        max_wraps=config.ge[Keys.MAX_WRAPS],
        max_recursion_depth=config.ge[Keys.MAX_RECURSION_DEPTH],
    )

    # Initialize Fitness Evaluator
    fitness_evaluator = FitnessEvaluator(fitness_functions=fitness_fn, mapper=mapper)

    # create Grammatical Evolution instance and run
    ge = GrammaticalEvolution(grammar=grammar, fitness_evaluator=fitness_evaluator)
    result = ge.run()

    print("Best Solution:", result.best_individual.phenotype)


if __name__ == "__main__":
    main()
