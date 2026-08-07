from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Union

import numpy as np
from tqdm import tqdm

from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.core.result import GEResult
from finchge.utils.logging_helpers import (
    IndividualLogHelper,
    LogIOHelper,
    ParetoFrontMetricsHelper,
    PopulationMetricsHelper,
    PopulationSamplingHelper,
)

LOG_DIRS = {}


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg: str = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(
    logger_id: str = "finch", group_name: Optional[str] = None, verbose: bool = False
) -> str:
    """
    Setup logging for a specific project instance.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = (
        f"logs/{group_name}/{logger_id}_{timestamp}"
        if group_name
        else f"logs/{logger_id}_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{logger_id}.log")
    LOG_DIRS[logger_id] = log_dir

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    root_logger.addHandler(fh)

    ch = TqdmLoggingHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    root_logger.addHandler(ch)

    logger = logging.getLogger(logger_id)
    logger.setLevel(logging.INFO)
    logger.propagate = True

    root_logger.info(f"Logging setup complete for {logger_id}. Log file: {log_file}")
    return log_dir


def get_logger(logger_id: str = "finch") -> logging.Logger:
    """
    Retrieve the logger for the given project instance.
    """
    return logging.getLogger(logger_id)


def get_log_dir(logger_id: str = "finch") -> str:
    """_summary_
                Retrieve the log directory for a given project instance.

    Args:
        logger_id (str, optional): _description_. Defaults to "finch".

    Raises:
        KeyError: _description_

    Returns:
        str: _description_
    """
    try:
        return LOG_DIRS[logger_id]
    except KeyError:
        raise KeyError(f"No log directory registered for logger_id '{logger_id}'")


class BaseLogger(ABC):
    """Interface for experiment tracking.
    Observer Pattern
    """

    @abstractmethod
    def on_run_start(
        self, log_dir: str, obj_names: list[str], config: dict[str, Any]
    ) -> None:
        """Called when a run starts."""
        pass

    @abstractmethod
    def on_generation_end(
        self,
        generation: int,
        population: Population,
        best: Optional[Individual] = None,
        pareto_front: Optional[list[Individual]] = None,
        fitness_stats: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Called at the end of each generation."""
        pass

    @abstractmethod
    def on_run_end(self, result: GEResult) -> None:
        """Called when a run ends."""
        pass


class ExperimentLogger(BaseLogger):
    """
    Experiment logger

    Logs essential data for post-hoc analysis.
    This provides minimal essential logs only.
    To add more logs, this class can be extended or wrapped to add custom metrics and analysis.
    """

    def __init__(
        self,
        exclude: Optional[Set[str]] = None,
        compress_genotypes: bool = False,
        log_population_samples: bool = False,
        sample_size: int = 5,
        custom_log_hook: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.exclude: Set[str] = exclude or set()
        self.compress_genotypes: bool = compress_genotypes
        self.log_population_samples: bool = log_population_samples
        self.sample_size: int = sample_size
        self.custom_log_hook: Optional[
            Callable[[dict[str, Any]], None]
        ] = custom_log_hook

        self.log_dir: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.objective_names: Optional[list[str]] = None

        self.generation_history: list[dict[str, Any]] = []
        self.run_start_time: Optional[datetime] = None

        self.algorithm: Optional[Any] = None

        self.generations_csv: Optional[Path] = None
        self.last_generation_time: Optional[datetime] = None

    def on_run_start(
        self,
        log_dir: str,
        obj_names: list[str],
        config: Dict[str, Any],
        algorithm: Optional[Any] = None,
    ) -> None:
        """
        Initialize logging when the experiment run starts.

        Args:
            algorithm: Optional reference to algorithm instance for custom logging hooks
        """

        self.log_dir = Path(log_dir)
        LogIOHelper.ensure_dir(self.log_dir)

        self.objective_names = obj_names
        self.algorithm = algorithm
        self.run_start_time = datetime.now()

        config_data = {
            "timestamp": self.run_start_time.isoformat(),
            "objective_names": obj_names,
            "n_objectives": len(obj_names),
            "config": config,
            "logger_settings": {
                "exclude": list(self.exclude),
                "compress_genotypes": self.compress_genotypes,
                "log_population_samples": self.log_population_samples,
                "sample_size": self.sample_size,
            },
        }

        LogIOHelper.write_json(self.log_dir / "experiment_config.json", config_data)

        self._init_dirs()
        self._init_generation_csv()
        self.last_generation_time = self.run_start_time

        logging.info(f"[ExperimentLogger] Started logging to: {log_dir}")
        logging.info(f"[ExperimentLogger] Objectives: {obj_names}")

    def _init_dirs(self) -> None:
        if not self.log_dir:
            raise AttributeError("Log directory is not set")

        if self.log_population_samples:
            LogIOHelper.ensure_dir(self.log_dir / "population_samples")

    def _get_generation_csv_headers(self) -> list[str]:
        if self.objective_names is None:
            raise AttributeError("Objective names are not set")
        headers = [
            "gen",
            "invalids",
            "total_inds",
            "unique_inds",
            "unused_search",
            "ave_genome_length",
            "max_genome_length",
            "min_genome_length",
            "ave_tree_depth",
            "max_tree_depth",
            "min_tree_depth",
            "ave_tree_nodes",
            "max_tree_nodes",
            "min_tree_nodes",
            "ave_used_codons",
            "max_used_codons",
            "min_used_codons",
            "time_taken",
            "total_time",
        ]

        if len(self.objective_names) == 1:
            headers[1:1] = [
                "ave_fitness",
                "best_fitness",
                "best_genome_length",
                "best_tree_depth",
                "best_tree_nodes",
                "best_used_codons",
                "best_phenotype_length",
            ]
        else:
            headers.insert(1, "front_size")

            for name in self.objective_names:
                headers.extend(
                    [
                        f"front_min_{name}",
                        f"front_max_{name}",
                        f"front_mean_{name}",
                    ]
                )

        return headers

    def _init_generation_csv(self) -> None:
        """Initialize the main generations CSV for the current run type."""
        if not self.log_dir:
            raise AttributeError("Log directory is not set")

        self.generations_csv = self.log_dir / "generations.csv"
        headers = self._get_generation_csv_headers()
        LogIOHelper.write_csv(self.generations_csv, headers, [])

    def on_generation_end(
        self,
        generation: int,
        population: Population,
        best: Optional[Individual] = None,
        pareto_front: Optional[list[Individual]] = None,
        fitness_stats: Optional[list[dict[str, Any]]] = None,
        extra_data: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log data at the end of each generation.

        Args:
            extra_data: Additional user-provided data to log
        """
        if self.log_dir is None:
            return

        history_entry = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(population),
        }

        if best is not None:
            self._log_single_objective(generation, best, fitness_stats, population)
            best_summary = IndividualLogHelper.build_summary(best)
            fitness = best_summary["fitness"]
            history_entry["best_fitness"] = fitness[0] if fitness else None
            history_entry["type"] = "single_objective"

        elif pareto_front is not None:
            self._log_multi_objective(generation, pareto_front, population)
            history_entry["front_size"] = len(pareto_front)
            history_entry["type"] = "multi_objective"

        if extra_data:
            history_entry["extra"] = extra_data

        self.generation_history.append(history_entry)

        if self.custom_log_hook:
            hook_data = {
                "generation": generation,
                "population": population,
                "best": best,
                "pareto_front": pareto_front,
                "fitness_stats": fitness_stats,
                "history_entry": history_entry,
                "algorithm": self.algorithm,
                "log_dir": self.log_dir,
            }
            try:
                self.custom_log_hook(hook_data)
            except Exception as e:
                logging.getLogger(__name__).warning("Custom logging hook failed: %s", e)

    def _log_single_objective(
        self,
        generation: int,
        best: Individual,
        fitness_stats: Optional[list[dict[str, Any]]],
        population: Population,
    ) -> None:
        """Log single-objective generation data."""
        if not self.log_dir:
            return

        IndividualLogHelper.write_individual_artifacts(
            base_dir=self.log_dir,
            ind=best,
            file_stem=str(generation),
            exclude=self.exclude,
            compress_genotypes=self.compress_genotypes,
        )

        metrics = PopulationMetricsHelper.compute(population)
        best_summary = IndividualLogHelper.build_summary(best)
        fitness = best_summary["fitness"]
        tree = IndividualLogHelper.get_tree(best)

        now = datetime.now()

        if self.last_generation_time is not None:
            time_taken = (now - self.last_generation_time).total_seconds()
        else:
            time_taken = 0.0

        if self.run_start_time is not None:
            total_time = (now - self.run_start_time).total_seconds()
        else:
            total_time = 0.0

        self.last_generation_time = now

        row = [
            generation,
            metrics["ave_fitness"],
            fitness[0] if fitness else None,
            best_summary["genotype_length"],
            best_summary["tree_depth"],
            PopulationMetricsHelper._tree_nodes(tree),
            best_summary["used_codons"],
            best_summary["phenotype_length"],
            metrics["invalids"],
            metrics["total_inds"],
            metrics["unique_inds"],
            metrics["unused_search"],
            metrics["ave_genome_length"],
            metrics["max_genome_length"],
            metrics["min_genome_length"],
            metrics["ave_tree_depth"],
            metrics["max_tree_depth"],
            metrics["min_tree_depth"],
            metrics["ave_tree_nodes"],
            metrics["max_tree_nodes"],
            metrics["min_tree_nodes"],
            metrics["ave_used_codons"],
            metrics["max_used_codons"],
            metrics["min_used_codons"],
            time_taken,
            total_time,
        ]

        if self.generations_csv:
            LogIOHelper.append_csv_row(self.generations_csv, row)

        if self.log_population_samples and population:
            self._log_population_sample(generation, population, "single_objective")

    def _log_multi_objective(
        self,
        generation: int,
        front: list[Individual],
        population: Population,
    ) -> None:
        """Log multi-objective generation data."""
        if not self.log_dir:
            return

        gen_dir = self.log_dir / f"generation_{generation}"
        LogIOHelper.ensure_dir(gen_dir)

        fitness_matrix = np.array([ind.fitness for ind in front])
        LogIOHelper.write_npy(gen_dir / "front_fitness.npy", fitness_matrix)

        rows: list[list[Any]] = []
        for i, ind in enumerate(front):
            ind_summary = IndividualLogHelper.build_summary(ind)
            tree = IndividualLogHelper.get_tree(ind)

            row: list[Any] = [i, *ind.fitness]
            row.extend(
                [
                    ind_summary["used_codons"],
                    ind_summary["genotype_length"],
                    ind_summary["tree_depth"],
                    PopulationMetricsHelper._tree_nodes(tree),
                    ind_summary["phenotype_length"],
                ]
            )
            rows.append(row)

            IndividualLogHelper.write_individual_artifacts(
                base_dir=gen_dir,
                ind=ind,
                file_stem=str(i),
                exclude=self.exclude,
                compress_genotypes=self.compress_genotypes,
            )

        if self.objective_names:
            headers = [
                "id",
                *self.objective_names,
                "used_codons",
                "genome_length",
                "tree_depth",
                "tree_nodes",
                "phenotype_length",
            ]
        else:
            n_objectives = len(front[0].fitness) if front else 0
            headers = [
                "id",
                *[f"obj{i}" for i in range(n_objectives)],
                "used_codons",
                "genome_length",
                "tree_depth",
                "tree_nodes",
                "phenotype_length",
            ]

        LogIOHelper.write_csv(gen_dir / "front.csv", headers, rows)

        front_metadata = {
            "generation": generation,
            "front_size": len(front),
            "timestamp": datetime.now().isoformat(),
            "objective_names": self.objective_names,
        }
        LogIOHelper.write_json(gen_dir / "metadata.json", front_metadata)

        now = datetime.now()

        if self.last_generation_time is not None:
            time_taken = (now - self.last_generation_time).total_seconds()
        else:
            time_taken = 0.0

        if self.run_start_time is not None:
            total_time = (now - self.run_start_time).total_seconds()
        else:
            total_time = 0.0

        self.last_generation_time = now

        pop_metrics = PopulationMetricsHelper.compute(population)
        front_metrics = ParetoFrontMetricsHelper.compute(front, self.objective_names)

        row_front: list[Any] = [
            generation,
            front_metrics.get("front_size"),
            pop_metrics.get("invalids"),
            pop_metrics.get("total_inds"),
            pop_metrics.get("unique_inds"),
            pop_metrics.get("unused_search"),
            pop_metrics.get("ave_genome_length"),
            pop_metrics.get("max_genome_length"),
            pop_metrics.get("min_genome_length"),
            pop_metrics.get("ave_tree_depth"),
            pop_metrics.get("max_tree_depth"),
            pop_metrics.get("min_tree_depth"),
            pop_metrics.get("ave_tree_nodes"),
            pop_metrics.get("max_tree_nodes"),
            pop_metrics.get("min_tree_nodes"),
            pop_metrics.get("ave_used_codons"),
            pop_metrics.get("max_used_codons"),
            pop_metrics.get("min_used_codons"),
            time_taken,
            total_time,
        ]

        if self.objective_names:
            for name in self.objective_names:
                row_front.extend(
                    [
                        front_metrics.get(f"front_min_{name}"),
                        front_metrics.get(f"front_max_{name}"),
                        front_metrics.get(f"front_mean_{name}"),
                    ]
                )
        elif front:
            n_objectives = len(front[0].fitness)
            for i in range(n_objectives):
                name = f"obj{i}"
                row_front.extend(
                    [
                        front_metrics.get(f"front_min_{name}"),
                        front_metrics.get(f"front_max_{name}"),
                        front_metrics.get(f"front_mean_{name}"),
                    ]
                )

        if self.generations_csv:
            LogIOHelper.append_csv_row(self.generations_csv, row_front)

        if self.log_population_samples and population:
            self._log_population_sample(generation, population, "multi_objective")

    def _log_population_sample(
        self,
        generation: int,
        population: Population,
        opt_type: str,
    ) -> None:
        """
        Save a sample of the full population for diversity analysis.
        """
        if not population:
            return

        if not self.log_dir:
            raise ValueError("Log directory not set.")

        sample_indices = PopulationSamplingHelper.sample_indices(
            population,
            self.sample_size,
        )
        sample = [population.individuals[i] for i in sample_indices]

        sample_dir = self.log_dir / "population_samples" / f"gen_{generation}"
        LogIOHelper.ensure_dir(sample_dir)

        sample_fitness = np.array([ind.fitness for ind in sample])
        LogIOHelper.write_npy(sample_dir / "sample_fitness.npy", sample_fitness)

        sample_rows: list[list[Any]] = []
        for idx, ind in zip(sample_indices, sample):
            summary = IndividualLogHelper.build_summary(ind)
            fitness = summary["fitness"]

            sample_rows.append(
                [
                    idx,
                    fitness,
                    summary["used_codons"],
                    summary["tree_depth"],
                    summary["phenotype_length"],
                    summary["genotype_length"],
                ]
            )

        LogIOHelper.write_csv(
            sample_dir / "sample_summary.csv",
            [
                "population_index",
                "fitness",
                "used_codons",
                "tree_depth",
                "phenotype_length",
                "genotype_length",
            ],
            sample_rows,
        )

        sample_metadata = {
            "generation": generation,
            "sample_size": len(sample),
            "total_population": len(population),
            "optimization_type": opt_type,
            "sample_indices": sample_indices,
            "timestamp": datetime.now().isoformat(),
        }
        LogIOHelper.write_json(sample_dir / "metadata.json", sample_metadata)

    def on_run_end(self, result: GEResult) -> None:
        """
        Finalize logging at the end of the run.
        """
        if self.log_dir is None:
            return

        run_end_time = datetime.now()
        run_duration = (
            (run_end_time - self.run_start_time).total_seconds()
            if self.run_start_time
            else 0
        )

        summary: Dict[str, Union[str, float, int, list[Any], None, Dict[str, Any]]] = {
            "run_start": (
                self.run_start_time.isoformat() if self.run_start_time else None
            ),
            "run_end": run_end_time.isoformat(),
            "run_duration_seconds": run_duration,
            "objective_names": self.objective_names,
            "n_objectives": len(self.objective_names) if self.objective_names else 0,
        }

        if result.all_time_best is not None:
            ind = result.all_time_best
            ind_summary = IndividualLogHelper.build_summary(ind)
            fitness = ind_summary["fitness"]

            summary["type"] = "single_objective"
            summary["best_fitness"] = fitness[0] if fitness else None
            summary["all_time_best"] = ind_summary

            payload = {
                "type": "single_objective",
                "fitness": fitness,
                "phenotype": IndividualLogHelper.get_phenotype(ind),
                "used_codons": ind_summary["used_codons"],
                "tree": IndividualLogHelper.get_tree(ind),
                "tree_depth": ind_summary["tree_depth"],
                "phenotype_length": ind_summary["phenotype_length"],
                "genotype_length": ind_summary["genotype_length"],
            }
            LogIOHelper.write_json(self.log_dir / "all_time_best.json", payload)

        elif result.pareto_front is not None:
            front = result.pareto_front
            summary["type"] = "multi_objective"
            summary["front_size"] = len(front)

            front_dir = self.log_dir / "final_pareto_front"
            LogIOHelper.ensure_dir(front_dir)

            fitness_matrix = np.array([ind.fitness for ind in front])
            LogIOHelper.write_npy(front_dir / "front_fitness.npy", fitness_matrix)

            rows: list[list[Any]] = []
            for i, ind in enumerate(front):
                ind_summary = IndividualLogHelper.build_summary(ind)
                row = [i, *ind.fitness]
                if ind_summary["used_codons"] is not None:
                    row.append(ind_summary["used_codons"])
                rows.append(row)

                IndividualLogHelper.write_individual_artifacts(
                    base_dir=front_dir,
                    ind=ind,
                    file_stem=str(i),
                    exclude=self.exclude,
                    compress_genotypes=self.compress_genotypes,
                )

            if self.objective_names:
                headers = ["id", *self.objective_names]
                if any(
                    IndividualLogHelper.get_used_codons(ind) is not None
                    for ind in front
                ):
                    headers.append("used_codons")

                LogIOHelper.write_csv(front_dir / "front.csv", headers, rows)

            front_metadata = {
                "front_size": len(front),
                "objective_names": self.objective_names,
                "timestamp": run_end_time.isoformat(),
            }
            LogIOHelper.write_json(front_dir / "metadata.json", front_metadata)

        LogIOHelper.write_json(self.log_dir / "run_summary.json", summary)

        logging.info(
            f"[ExperimentLogger] Run completed. Results saved to: {self.log_dir}"
        )
        logging.info(f"[ExperimentLogger] Run duration: {run_duration:.2f} seconds")
