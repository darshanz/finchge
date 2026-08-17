import json
import tempfile
from pathlib import Path

from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.core.result import GEResult
from finchge.utils.logger import ExperimentLogger


def _make_individual(fitness_value: float) -> Individual:
    ind = Individual.from_genotype([1, 2, 3])
    ind.phenotype = "x + 1"
    ind.fitness = [fitness_value]
    ind.used_codon_count = 3
    return ind


def _make_population(n: int = 4) -> Population:
    inds = [_make_individual(float(i)) for i in range(1, n + 1)]
    return Population.from_individuals(inds, population_size=n)


def test_on_generation_end_does_not_crash():
    logger = ExperimentLogger()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.on_run_start(tmpdir, ["fitness"], config={})
        pop = _make_population()
        best = _make_individual(1.0)
        logger.on_generation_end(generation=1, population=pop, best=best)


def test_on_run_end_writes_all_time_best_json():
    """Regression: on_run_end must use result.all_time_best, not best_in_generation."""
    logger = ExperimentLogger()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.on_run_start(tmpdir, ["fitness"], config={})
        pop = _make_population()
        best = _make_individual(1.0)
        logger.on_generation_end(generation=1, population=pop, best=best)

        all_time = _make_individual(0.5)
        last_gen_best = _make_individual(1.5)
        result = GEResult(
            all_time_best=all_time,
            best_in_generation=last_gen_best,
            pareto_front=None,
            population=pop,
        )
        logger.on_run_end(result)

        summary_path = Path(tmpdir) / "run_summary.json"
        assert summary_path.exists(), "run_summary.json was not written"
        with open(summary_path) as f:
            data = json.load(f)

        # The logged best fitness must come from all_time_best (0.5), not best_in_generation (1.5)
        assert data["best_fitness"] == 0.5


def test_on_run_end_json_contains_all_time_best_key():
    """Regression: run_summary.json must have 'all_time_best' key."""
    logger = ExperimentLogger()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.on_run_start(tmpdir, ["fitness"], config={})
        pop = _make_population()
        logger.on_generation_end(generation=1, population=pop)

        best = _make_individual(1.0)
        result = GEResult(
            all_time_best=best,
            best_in_generation=best,
            pareto_front=None,
            population=pop,
        )
        logger.on_run_end(result)

        with open(Path(tmpdir) / "run_summary.json") as f:
            data = json.load(f)

        assert "all_time_best" in data


def test_logger_generation_csv_is_created():
    logger = ExperimentLogger()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.on_run_start(tmpdir, ["fitness"], config={})
        assert (Path(tmpdir) / "generations.csv").exists()


def test_experiment_config_json_is_written_on_start():
    logger = ExperimentLogger()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.on_run_start(tmpdir, ["fitness"], config={"key": "value"})
        config_path = Path(tmpdir) / "experiment_config.json"
        assert config_path.exists()
        with open(config_path) as f:
            data = json.load(f)
        assert data["config"] == {"key": "value"}
