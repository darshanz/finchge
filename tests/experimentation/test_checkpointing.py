import random

import pytest

from finchge.fitness.fitness_functions import GEFitnessFunction
from finchge.fitness.fitness_types import Fitness
from finchge.grammar import GenotypeMapper, Grammar
from finchge.utils.checkpoint import FileCheckpointManager, stable_config_hash
from finchge.utils.logger import ExperimentLogger


@pytest.fixture(autouse=True)
def seed_rng():
    seed = 42
    random.seed(seed)


@pytest.fixture
def simple_grammar():
    grammar_str = """
    <expr> ::= <expr> <op> <expr> | <var> | <const>
    <op> ::= + | -
    <var> ::= x
    <const> ::= 1.0 | 2.0
    """
    return Grammar(grammar_str)


class LengthFitness(GEFitnessFunction):
    def __init__(self) -> None:
        super().__init__(maximize=False)

    def evaluate(self, phenotype: str):
        # Deterministic, cheap
        return Fitness(len(phenotype))


@pytest.fixture
def fitness_evaluator(simple_grammar):
    from finchge.fitness.fitness_evaluator import FitnessEvaluator

    mapper = GenotypeMapper(grammar=simple_grammar, random_state=1234)

    fitness_eval = FitnessEvaluator(fitness_functions=LengthFitness(), mapper=mapper)
    return fitness_eval


@pytest.fixture
def ge_config():
    return {
        "experiment": {
            "num_generations": 10,
            "verbose": True,
            "random_seed": 1234,
            "cache_type": "lru",
            "cache_size": 128,
        },
        "ge": {
            "population_size": 100,
            "grammar_file": "grammar.bnf",
            "codon_size": 127,
            "max_wraps": 6,
            "max_recursion_depth": 10,
            "genome_length": 100,
            "mutation_probability": 0.01,
            "crossover_probability": 0.5,
            "elite_size": 3,
            "init_type": "random_genome",
        },
    }


@pytest.fixture
def checkpoint_manager(tmp_path):
    return FileCheckpointManager(
        directory=tmp_path / "checkpoints",
        every=5,
        keep_last=3,
    )


@pytest.fixture
def ge_factory(simple_grammar, fitness_evaluator, ge_config):
    def _make(
        *,
        checkpoint_manager=None,
    ):
        cfg = dict(ge_config)

        from finchge.core.engine import GrammaticalEvolution

        return GrammaticalEvolution(
            grammar=simple_grammar,
            fitness_evaluator=fitness_evaluator,
            config=cfg,
            checkpoint_manager=checkpoint_manager,
            expt_logger=ExperimentLogger(),
        )

    return _make


def test_checkpoint_resume_equivalence(
    ge_factory,
    tmp_path,
):
    """
    Running evolution fully vs. stopping halfway and resuming
    from a checkpoint must produce identical results.

    Run evolution fully,
    Run evolution - stop after 5 genrations ,
    then run from the 5th generation checkpoint.
    Both full run and reumed run should have same results as running fully

    """

    checkpoint_dir = tmp_path / "checkpoints"

    checkpoint_manager = FileCheckpointManager(
        directory=checkpoint_dir,
        every=2,
        keep_last=3,
    )

    #  Full uninterrrupted run
    ge_full = ge_factory()
    result_full = ge_full.run()

    #  this time interrupt in the middle
    ge_partial = ge_factory(checkpoint_manager=checkpoint_manager)

    # stop in 30s
    import threading
    import time

    def stopper():
        time.sleep(0.05)
        ge_partial.halt(min_gens_allowed=5)  # halt after 5 generatons

    threading.Thread(target=stopper, daemon=True).start()
    ge_partial.run()

    #  Then continue ...
    ge_resume = ge_factory(checkpoint_manager=checkpoint_manager)
    result_resumed = ge_resume.run()

    #  then check
    assert result_full.best_individual.fitness == result_resumed.best_individual.fitness
    assert (
        result_full.best_individual.phenotype
        == result_resumed.best_individual.phenotype
    )


def test_rng_state_saved(checkpoint_manager, ge_factory):
    ge = ge_factory(checkpoint_manager=checkpoint_manager)
    ge.run()

    # Load checkpoint state
    checkpoint = checkpoint_manager.load_latest()
    assert checkpoint is not None
    assert hasattr(checkpoint, "rng_state")


def test_rng_state_restore(checkpoint_manager, ge_factory):
    # Run once and checkpoint
    ge = ge_factory(checkpoint_manager=checkpoint_manager)
    ge.run()

    checkpoint = checkpoint_manager.load_latest()
    assert checkpoint is not None
    assert checkpoint.rng_state is not None

    # Restore RNG state
    random.setstate(checkpoint.rng_state.py_state)

    # Generate future sequence
    seq1 = [random.random() for _ in range(10)]

    # Restore again
    random.setstate(checkpoint.rng_state.py_state)

    # Generate again
    seq2 = [random.random() for _ in range(10)]

    assert seq1 == seq2


def test_checkpoint_config_mismatch(checkpoint_manager, ge_factory):
    ge = ge_factory(checkpoint_manager=checkpoint_manager)
    ge.run()

    # Copy the real config and mutate it
    bad_config = ge.config.copy({"experiment": {"population_size": 999}})

    bad_hash = stable_config_hash(bad_config)

    # Should raise Runtime Error with following message
    with pytest.raises(
        RuntimeError,
        match="Checkpoint config hash does not match current config."
        " Refusing to resume to prevent inconsistent runs.",
    ):
        checkpoint_manager.load_latest(expected_config_hash=bad_hash)


def test_checkpoint_config_match(checkpoint_manager, ge_factory):
    ge = ge_factory(checkpoint_manager=checkpoint_manager)
    ge.run()

    import copy

    assert stable_config_hash(ge.config) == stable_config_hash(copy.deepcopy(ge.config))


def test_checkpoint_manager_creates_directory(tmp_path):
    checkpoint_dir = tmp_path / "new_checkpoints"
    assert not checkpoint_dir.exists()

    FileCheckpointManager(
        directory=checkpoint_dir,
        every=1,
        keep_last=1,
    )

    # Directory should be created
    assert checkpoint_dir.exists()
