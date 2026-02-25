from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finchge.core.individual import Individual

import csv
import math
import statistics
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from finchge.utils.logger import get_log_dir, get_logger
from finchge.utils.visualization import (
    plot_best_fitness,
    plot_tree,
    visualize_pareto_front,
)


class ResultHelper:
    def __init__(self, project_id: str = "finch") -> None:
        self.project_id = project_id
        self.logger = get_logger(project_id)

    def generate_summary(self) -> None:
        """
        Generate and save summary statistics for single-objective optimization utils

        Args:
            objective_names (list): Names of the objective functions
        """

        csv_file_path = f"{get_log_dir(self.project_id)}/generations.csv"
        try:
            generation_data = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            self.logger.info(
                f"Skipped generation summary. CSV file not found for generating summary. "
                f" {csv_file_path}   To automatically generate summary enable ExperimentLogger."
            )
            return

        def get_column_stats(
            column_name: str,
        ) -> tuple[float, float, float, float] | None:
            """
            Get statistics for a column if it exists, otherwise return None
            Some colums may not exist, stats will show None for such columns. For example,
            For Tree based work flow, the used_codons may not be saved, as no integer genome is involved.
            """
            if column_name not in generation_data.columns:
                self.logger.warning(
                    f"Column '{column_name}' not found in CSV. Skipping."
                )
                return None

            col_data = generation_data[column_name]
            # Handle empty columns or columns with all NaN values
            if col_data.empty or col_data.isna().all():
                self.logger.warning(
                    f"Column '{column_name}' has no valid data. Skipping."
                )
                return None

            # Drop NaN values for calculation
            clean_data = col_data.dropna()
            if len(clean_data) == 0:
                return None

            max_ = float(round(clean_data.max(), 4))
            min_ = float(round(clean_data.min(), 4))
            avg_ = round(statistics.mean(clean_data), 4)
            std_ = round(statistics.stdev(clean_data), 4)
            return min_, max_, avg_, std_

        # Prepare tables with optional columns
        headers = ["Min", "Max", "Average(±std)"]
        fitness_stats = []

        # Always include best_fitness if available
        bf = get_column_stats("best_fitness")
        if bf:
            fitness_stats.append(["Best Fitness", bf[0], bf[1], f"{bf[2]}(±{bf[3]})"])

        # Optional metrics - only include if data exists
        optional_metrics = [
            ("valid_count", "Valid Individuals"),
            ("used_codons", "Used Codons"),
            ("tree_depth", "Tree Depth"),
        ]

        for col_name, display_name in optional_metrics:
            stats = get_column_stats(col_name)
            if stats:
                fitness_stats.append(
                    [display_name, stats[0], stats[1], f"{stats[2]}(±{stats[3]})"]
                )
            else:
                fitness_stats.append([display_name, "N/A", "N/A", "N/A"])

        # Check if there are any statistics to display
        if not fitness_stats:
            self.logger.error(
                "No valid statistics could be calculated from the CSV data."
            )
            return

        summary = "\nStatistics:\n"
        summary += tabulate(fitness_stats, headers=headers, tablefmt="pretty")

        self.logger.info(summary)
        summary_path = f"{get_log_dir(self.project_id)}/summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        self.logger.info(f"Summary saved to {summary_path}")

        # Plot fitness chart
        try:
            plot_best_fitness(
                best_fitness=generation_data["best_fitness"].to_numpy(dtype=np.float64),
                save_path=f"{get_log_dir(self.project_id)}/fitness_chart.png",
            )
            self.logger.info(
                f"Fitness chart saved to {get_log_dir(self.project_id)}/fitness_chart.png"
            )
        except Exception as e:
            self.logger.error(f"Exception {e}")

        # Assuming Eliticism.. finchGE is built with eliticism
        # If changed later this may need to be updated by
        # perhaps the tree of the best individual can be saved at the end of the evolution
        best_generation = generation_data["generation"].max()
        self.logger.info(
            f"Tree saved for: {best_generation} --  {get_log_dir(self.project_id)}/trees/{best_generation}.json"
        )

        # tree for best generation

        best_tree_path = f"{get_log_dir(self.project_id)}/trees/{best_generation}.json"
        try:
            with open(best_tree_path, "r") as file:
                best_tree_json = file.read()
            plot_tree(best_tree_json, get_log_dir(self.project_id))
        except FileNotFoundError:
            self.logger.info(
                f"Skipping plotting phenotype mapping tree: Tree file for generation {best_generation} not found. Please ensure logging is not excluded for tree files in config."
            )

    def save_pareto_front(
        self, pareto_front: list["Individual"], objective_names: list[str]
    ) -> None:
        pareto_csv_path = f"{get_log_dir()}/pareto_front.csv"

        # fitness objectives + phenotype
        headers = objective_names + ["phenotype"]

        with open(pareto_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for ind in pareto_front:
                if ind.fitness is None:
                    fitness = []
                elif isinstance(ind.fitness, list):
                    fitness = ind.fitness
                else:
                    fitness = [ind.fitness]

                row = fitness + [ind.phenotype]
                writer.writerow(row)

        save_dir = get_log_dir(self.project_id)
        visualize_pareto_front(save_dir, objective_names)


class StatsHelper:
    """
    Utility class for computing population-level fitness statistics.
    """

    @staticmethod
    def compute_fitness_stats(
        *,
        individuals: list["Individual"],
        objective_names: Optional[list[str]] = None,
    ) -> list[Dict[str, Any]]:
        """
        Compute fitness statistics for a population.

        Args:
            individuals:
                List of evaluated Individuals.
            objective_names:
                Optional list of objective names.
                If None, objectives will be indexed numerically.

        Returns:
            A list of dictionaries, one per objective, containing:
            - fitness_name
            - min
            - max
            - avg
            - std
            - total_count
            - valid_count
        """

        if not individuals:
            return []

        # Determine number of objectives from first valid individual
        first_fitness = next(
            (ind.fitness for ind in individuals if ind.fitness is not None),
            None,
        )

        if first_fitness is None:
            return []

        num_objectives = len(first_fitness)

        if objective_names is None:
            objective_names = [f"objective_{i}" for i in range(num_objectives)]
        else:
            if len(objective_names) != num_objectives:
                raise ValueError(
                    "Length of objective_names does not match fitness dimensionality"
                )

        stats: list[Dict[str, Any]] = []
        total_count = len(individuals)

        for obj_idx in range(num_objectives):
            values: list[float] = []

            for ind in individuals:
                if ind.fitness is None:
                    continue

                val = ind.fitness[obj_idx]

                if val is None:
                    continue
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    continue

                values.append(val)

            if not values:
                stats.append(
                    {
                        "fitness_name": objective_names[obj_idx],
                        "min": None,
                        "max": None,
                        "avg": None,
                        "std": None,
                        "total_count": total_count,
                        "valid_count": 0,
                    }
                )
                continue

            stats.append(
                {
                    "fitness_name": objective_names[obj_idx],
                    "min": min(values),
                    "max": max(values),
                    "avg": statistics.mean(values),
                    "std": (statistics.stdev(values) if len(values) > 1 else 0.0),
                    "total_count": total_count,
                    "valid_count": len(values),
                }
            )

        return stats
