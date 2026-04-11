from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, TextIO, Union

import numpy as np
from tqdm import tqdm

from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.core.result import GEResult
from finchge.grammar.derivation_tree import TreeNode

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


class FileLogger(BaseLogger):
    def __init__(self) -> None:
        self.run_dir: Optional[Path] = None
        self.metrics_file: Optional[TextIO] = None

    def on_run_start(
        self, log_dir: str, obj_names: list[str], config: dict[str, Any]
    ) -> None:
        self.run_dir = Path(log_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        metrics_path = self.run_dir / "metrics.jsonl"
        self.metrics_file = open(metrics_path, "a")

    def on_generation_end(
        self,
        generation: int,
        population: Population,
        best: Optional[Individual] = None,
        pareto_front: Optional[list[Individual]] = None,
        fitness_stats: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if best is not None:
            record = {
                "generation": generation,
                "fitness": best.fitness,
                "phenotype": best.phenotype,
            }

        elif pareto_front is not None:
            fitness = np.array([ind.fitness for ind in pareto_front])

            record = {
                "generation": generation,
                "front_size": len(pareto_front),
                "fitness_min": fitness.min(axis=0).tolist(),
                "fitness_max": fitness.max(axis=0).tolist(),
            }

        else:
            return
        if self.metrics_file:
            self.metrics_file.write(json.dumps(record) + "\n")
            self.metrics_file.flush()

    def on_run_end(self, result: GEResult) -> None:
        if self.run_dir is None:
            return

        if result.best_individual is not None:
            payload: dict[str, Union[list[float], str, int, None]] = {
                "type": "single_objective",
                "fitness": result.best_individual.fitness,
                "phenotype": result.best_individual.phenotype,
            }

        elif result.pareto_front is not None:
            fitness = np.array([ind.fitness for ind in result.pareto_front])
            payload = {
                "type": "multi_objective",
                "front_size": len(result.pareto_front),
                "fitness_min": fitness.min(axis=0).tolist(),
                "fitness_max": fitness.max(axis=0).tolist(),
            }

        else:
            payload = {"type": "empty"}

        with open(self.run_dir / "final_summary.json", "w") as f:
            json.dump(payload, f, indent=2)

        if self.metrics_file:
            self.metrics_file.close()


class ExperimentLogger(FileLogger):
    """
    Experiment logger

     Logs essential data for post-hoc analysis.
     This provided minimal essential logs only.
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
        """
        Logs essential data for post-hoc analysis.

        Args:
            exclude: Fields to exclude from logging ('phenotypes', 'genotypes', 'trees')
            compress_genotypes: Save genotypes as compressed numpy arrays
            log_population_samples: Save samples from full population (not just front/best)
            sample_size: Number of individuals to sample from population
            custom_log_hook: Function to call with extra logging data each generation
        """
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

        # history for analysis
        self.generation_history: list[dict[str, Any]] = []
        self.run_start_time: Optional[datetime] = None

        # reference to algorithm for custom logging
        self.algorithm: Optional[Any] = None

    def on_run_start(
        self,
        log_dir: str,
        obj_names: list[str],
        config: Dict[str, Any],
        algorithm: Optional[Any] = None,
    ) -> None:
        """
        Initialize logging when the experiment run starts.
        Mainly creates different directories for logging, prepares config data and expt metadata for logging
        `log_dir` should be provided to store logs. Not autogenerated to maintain consistency.


        Args:
            algorithm: Optional reference to algorithm instance for custom logging hooks
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.objective_names = obj_names
        self.algorithm = algorithm
        self.run_start_time = datetime.now()

        # Save config with timestamp and metadata
        config_data = {
            "timestamp": self.run_start_time.isoformat(),
            "objective_names": obj_names,
            "n_objectives": len(obj_names),
            "config": config,
            "logger_settings": {
                "exclude": list(self.exclude),
                "compress_genotypes": self.compress_genotypes,
                "log_population_samples": self.log_population_samples,
            },
        }

        config_path = self.log_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        # Initialize directories
        self._init_dirs()

        # Initialize CSV for single-objective
        if len(obj_names) == 1:
            self._init_single_objective_csv()

        print(f"[ExperimentLogger] Started logging to: {log_dir}")
        print(f"[ExperimentLogger] Objectives: {obj_names}")

    def _init_dirs(self) -> None:
        if not self.log_dir:
            raise AttributeError("Log directory is not set")

        # Create population samples directory if enabled
        if self.log_population_samples:
            (self.log_dir / "population_samples").mkdir(exist_ok=True)

    def _init_single_objective_csv(self) -> None:
        if not self.log_dir:
            raise AttributeError("Log directory is not set")

        self.csv_path = self.log_dir / "generations.csv"
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            stat_cols = ["min", "max", "avg", "std", "total_count", "valid_count"]
            writer.writerow(
                ["generation", "best_fitness", *stat_cols, "used_codons", "tree_depth"]
            )

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

        # Store minimal history in memory
        history_entry = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(population),
        }

        if best is not None:
            # Single-objective logging
            self._log_single_objective(generation, best, fitness_stats, population)
            history_entry["best_fitness"] = best.fitness[0]
            history_entry["type"] = "single_objective"

        elif pareto_front is not None:
            # Multi-objective logging
            self._log_multi_objective(generation, pareto_front, population)
            history_entry["front_size"] = len(pareto_front)
            history_entry["type"] = "multi_objective"

        # Add extra data to history
        if extra_data:
            history_entry["extra"] = extra_data

        # Store in history list
        self.generation_history.append(history_entry)

        # Call custom logging hook if provided
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

        # Create phenotype/genotype/tree directories if not excluded
        if "phenotypes" not in self.exclude:
            (self.log_dir / "phenotypes").mkdir(exist_ok=True)
        if "genotypes" not in self.exclude:
            (self.log_dir / "genotypes").mkdir(exist_ok=True)
        if "trees" not in self.exclude:
            (self.log_dir / "trees").mkdir(exist_ok=True)

        if best.phenotype and "phenotypes" not in self.exclude:
            phenotype_path = self.log_dir / "phenotypes" / f"{generation}.txt"
            phenotype_path.write_text(best.phenotype)

        if "genotypes" not in self.exclude and hasattr(best, "genotype"):
            genotype_path = self.log_dir / "genotypes" / f"{generation}.txt"
            genotype_path.write_text(str(best.genotype))

        if best.tree and "trees" not in self.exclude:
            tree_path = self.log_dir / "trees" / f"{generation}_tree.txt"
            tree_path.write_text(best.tree)

        # calculate tree depth
        depth = 0
        if best.tree:
            try:
                depth = TreeNode.from_string(best.tree).max_depth
            except (AttributeError, ValueError, TypeError):
                depth = 0

        # write to CSV
        if self.csv_path and fitness_stats:
            fitness_stats_dict = fitness_stats[0]  # single objective only
            stat_cols = ["min", "max", "avg", "std", "total_count", "valid_count"]
            stats_values = [fitness_stats_dict[col] for col in stat_cols]

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        generation,
                        best.fitness[0],
                        *stats_values,
                        best.used_codon_count if hasattr(best, "used_codons") else 0,
                        depth,
                    ]
                )

        # Log population samples if enabled
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
        gen_dir.mkdir(exist_ok=True)

        # Save front fitness matrix as npy
        fitness_matrix = np.array([ind.fitness for ind in front])
        np.save(gen_dir / "front_fitness.npy", fitness_matrix)

        # Save as CSV for easy viewing
        rows: list[list[Union[float, int]]] = []
        for i, ind in enumerate(front):
            row: list[Union[float, int]] = [i, *ind.fitness]
            if hasattr(ind, "used_codons"):
                row.append(ind.used_codon_count)
            rows.append(row)

            # Save individual files
            if ind.phenotype and "phenotypes" not in self.exclude:
                (gen_dir / "phenotypes").mkdir(exist_ok=True)
                (gen_dir / "phenotypes" / f"{i}.txt").write_text(ind.phenotype)

            if "genotypes" not in self.exclude and hasattr(ind, "genotype"):
                (gen_dir / "genotypes").mkdir(exist_ok=True)
                if ind.genotype:
                    if self.compress_genotypes:
                        np.savez_compressed(
                            gen_dir / "genotypes" / f"{i}.npz", genotype=ind.genotype
                        )
                    else:
                        (gen_dir / "genotypes" / f"{i}.txt").write_text(
                            str(ind.genotype) if ind.genotype else ""
                        )

            if ind.tree and "trees" not in self.exclude:
                (gen_dir / "trees").mkdir(exist_ok=True)
                (gen_dir / "trees" / f"{i}_tree.txt").write_text(ind.tree)

        # Save CSV with headers
        if self.objective_names:
            headers = ["id", *self.objective_names]
            if any(hasattr(ind, "used_codons") for ind in front):
                headers.append("used_codons")

            with open(gen_dir / "front.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

        # Save front metadata
        front_metadata = {
            "generation": generation,
            "front_size": len(front),
            "timestamp": datetime.now().isoformat(),
            "objective_names": self.objective_names,
        }
        with open(gen_dir / "metadata.json", "w") as f:
            json.dump(front_metadata, f, indent=2)

        # Log population samples if enabled
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

        # Sample individuals (stratified by fitness)
        sample_indices = self._sample_population_indices(population, self.sample_size)
        sample = [population.individuals[i] for i in sample_indices]

        sample_dir = self.log_dir / "population_samples" / f"gen_{generation}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save sample fitness
        sample_fitness = np.array([ind.fitness for ind in sample])
        np.save(sample_dir / "sample_fitness.npy", sample_fitness)

        # Save sample metadata
        sample_metadata = {
            "generation": generation,
            "sample_size": len(sample),
            "total_population": len(population),
            "optimization_type": opt_type,
            "sample_indices": sample_indices,
            "timestamp": datetime.now().isoformat(),
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(sample_metadata, f, indent=2)

    def _sample_population_indices(
        self,
        population: Population,
        sample_size: int,
    ) -> list[int]:
        """
        Sample indices from population, trying to capture diversity.
        """
        n = len(population)
        sample_size = min(sample_size, n)

        if n <= sample_size:
            return list(range(n))

        # Simple stratified sampling by fitness percentiles
        try:
            # For single-objective or first objective
            fitness_values = [ind.fitness[0] for ind in population.individuals]
            sorted_indices = np.argsort(fitness_values)
            # Take from different parts of the distribution
            indices = []
            for i in range(sample_size):
                pos = int(i * (n - 1) / (sample_size - 1)) if sample_size > 1 else 0
                indices.append(sorted_indices[pos])
            return indices
        except (IndexError, ValueError):
            # Fallback: linear sampling
            if sample_size == 0:
                return []
            step = n / sample_size
            return [int(i * step) for i in range(sample_size)]

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

        # Save run summary
        summary: Dict[str, Union[str, float, int, list[Any], None, Dict[str, Any]]] = {
            "run_start": (
                self.run_start_time.isoformat() if self.run_start_time else None
            ),
            "run_end": run_end_time.isoformat(),
            "run_duration_seconds": run_duration,
            "objective_names": self.objective_names,
            "n_objectives": len(self.objective_names) if self.objective_names else 0,
        }

        # Add result-specific data
        if result.best_individual is not None:
            ind = result.best_individual
            summary["type"] = "single_objective"
            summary["best_fitness"] = ind.fitness[0] if ind.fitness else None
            summary["best_individual"] = {
                "fitness": ind.fitness[0],
                "used_codons": getattr(ind, "used_codons", None),
                "has_phenotype": ind.phenotype is not None,
                "has_tree": ind.tree is not None,
            }

            # Save best individual
            payload = {
                "type": "single_objective",
                "fitness": ind.fitness,
                "phenotype": ind.phenotype,
                "used_codons": ind.used_codon_count,
                "tree": ind.tree,
            }
            with open(self.log_dir / "best_individual.json", "w") as f:
                json.dump(payload, f, indent=2, default=str)

        elif result.pareto_front is not None:
            front = result.pareto_front
            summary["type"] = "multi_objective"
            summary["front_size"] = len(front)

            # Save final Pareto front
            front_dir = self.log_dir / "final_pareto_front"
            front_dir.mkdir(exist_ok=True)

            # Save fitness matrix
            fitness_matrix = np.array([ind.fitness for ind in front])
            np.save(front_dir / "front_fitness.npy", fitness_matrix)

            # Save individual files
            rows = []
            for i, ind in enumerate(front):
                row = [i, *ind.fitness]
                if hasattr(ind, "used_codons"):
                    row.append(ind.used_codon_count)
                rows.append(row)

                (front_dir / f"{i}_phenotype.txt").write_text(ind.phenotype or "")

                if self.compress_genotypes and hasattr(ind, "genotype"):
                    if ind.genotype:
                        np.savez_compressed(
                            front_dir / f"{i}_genotype.npz", genotype=ind.genotype
                        )
                elif hasattr(ind, "genotype"):
                    (front_dir / f"{i}_genotype.txt").write_text(str(ind.genotype))

                if ind.tree:
                    (front_dir / f"{i}_tree.json").write_text(ind.tree)

            # Save CSV
            if self.objective_names:
                headers = ["id", *self.objective_names]
                if any(hasattr(ind, "used_codons") for ind in front):
                    headers.append("used_codons")

                with open(front_dir / "front.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(rows)

            # Save front metadata
            front_metadata = {
                "front_size": len(front),
                "objective_names": self.objective_names,
                "timestamp": run_end_time.isoformat(),
            }
            with open(front_dir / "metadata.json", "w") as f:
                json.dump(front_metadata, f, indent=2)

        # Save generation history
        if self.generation_history:
            history_path = self.log_dir / "generation_history.json"
            with open(history_path, "w") as f:
                json.dump(self.generation_history, f, indent=2, default=str)

        # Save final summary
        summary_path = self.log_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"[ExperimentLogger] Run completed. Results saved to: {self.log_dir}")
        print(f"[ExperimentLogger] Run duration: {run_duration:.2f} seconds")
