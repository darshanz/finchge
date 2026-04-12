from pathlib import Path
from typing import TYPE_CHECKING

from finchge.grammar.derivation_tree import TreeNode

if TYPE_CHECKING:
    from finchge.core.individual import Individual

import csv
import math
import statistics
from typing import Any, Dict, Optional

import pandas as pd
from tabulate import tabulate

from finchge.utils.logger import get_log_dir, get_logger
from finchge.utils.visualization import (
    plot_front_size,
    plot_generation_runtime,
    plot_objective_progress,
    plot_search_diagnostics,
    plot_single_objective_complexity,
    plot_single_objective_fitness,
    plot_tree,
    visualize_final_pareto_front,
    visualize_pareto_front_evolution,
)


class ResultHelper:
    def __init__(self, project_id: str = "finch") -> None:
        self.project_id = project_id
        self.logger = get_logger(project_id)

    def generate_summary(self, objective_names: Optional[list[str]] = None) -> None:
        csv_file_path = f"{get_log_dir(self.project_id)}/generations.csv"

        try:
            generation_data = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            self.logger.info(
                f"Skipped generation summary. CSV file not found: {csv_file_path}. "
                f"Enable ExperimentLogger to generate it."
            )
            return

        if generation_data.empty:
            self.logger.info("Skipped generation summary. generations.csv is empty.")
            return

        if objective_names is None:
            raise ValueError("objective_names is required for multi-objective summary")

        if len(objective_names) > 1:
            self._generate_multiobjective_summary(generation_data, objective_names)
        else:
            self._generate_single_objective_summary(generation_data)

    def _generate_single_objective_summary(
        self,
        generation_data: pd.DataFrame,
    ) -> None:
        """
        Generate and save summary statistics from generations.csv.
        """

        def get_column_stats(
            column_name: str,
        ) -> tuple[float, float, float, float] | None:
            if column_name not in generation_data.columns:
                return None

            col_data = pd.to_numeric(
                generation_data[column_name], errors="coerce"
            ).dropna()
            if len(col_data) == 0:
                return None

            min_ = float(round(col_data.min(), 4))
            max_ = float(round(col_data.max(), 4))
            avg_ = float(round(col_data.mean(), 4))
            std_ = float(round(col_data.std(ddof=1), 4)) if len(col_data) > 1 else 0.0
            return min_, max_, avg_, std_

        summary_metrics = [
            ("best_fitness", "Best Fitness"),
            ("ave_fitness", "Average Fitness"),
            ("invalids", "Invalid Individuals"),
            ("unique_inds", "Unique Individuals"),
            ("unused_search", "Unused Search (%)"),
            ("ave_used_codons", "Average Used Codons"),
            ("best_used_codons", "Best Used Codons"),
            ("ave_tree_depth", "Average Tree Depth"),
            ("best_tree_depth", "Best Tree Depth"),
            ("ave_tree_nodes", "Average Tree Nodes"),
            ("best_tree_nodes", "Best Tree Nodes"),
            ("ave_genome_length", "Average Genome Length"),
            ("best_genome_length", "Best Genome Length"),
            ("time_taken", "Time Per Generation"),
            ("total_time", "Total Time"),
        ]

        headers = ["Metric", "Min", "Max", "Average(±std)"]
        summary_rows = []

        for col_name, display_name in summary_metrics:
            stats = get_column_stats(col_name)
            if stats is None:
                continue
            summary_rows.append(
                [display_name, stats[0], stats[1], f"{stats[2]}(±{stats[3]})"]
            )

        if not summary_rows:
            self.logger.error(
                "No valid statistics could be calculated from generations.csv."
            )
            return

        summary = "\nStatistics:\n"
        summary += tabulate(summary_rows, headers=headers, tablefmt="pretty")

        self.logger.info(summary)

        summary_path = f"{get_log_dir(self.project_id)}/summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)

        self.logger.info(f"Summary saved to {summary_path}")

        # Plot best tree from actual best generation
        if (
            "best_fitness" in generation_data.columns
            and "gen" in generation_data.columns
        ):
            try:
                best_row = generation_data.loc[
                    pd.to_numeric(
                        generation_data["best_fitness"], errors="coerce"
                    ).idxmin()
                ]
                best_generation = int(best_row["gen"])

                best_tree_path = (
                    f"{get_log_dir(self.project_id)}/trees/{best_generation}_tree.txt"
                )

                self.logger.info(
                    f"Tree saved for best generation: {best_generation} -- {best_tree_path}"
                )

                with open(best_tree_path, "r") as file:
                    best_tree = file.read()

                plot_tree(best_tree, get_log_dir(self.project_id))

            except FileNotFoundError:
                self.logger.info(
                    "Skipping plotting phenotype mapping tree: tree file not found for "
                    "best generation. Ensure tree logging is enabled."
                )
            except Exception as e:
                self.logger.error(f"Exception while plotting best tree: {e}")

            # Plots
        self.plot_single_objective_diagnostics()

    def _generate_multiobjective_summary(
        self,
        generation_data: pd.DataFrame,
        objective_names: list[str],
    ) -> None:
        save_dir = get_log_dir(self.project_id)
        pareto_csv_path = f"{save_dir}/final_pareto_front/front.csv"

        try:
            pareto_data = pd.read_csv(pareto_csv_path)
        except FileNotFoundError:
            self.logger.info(
                f"Skipped Pareto front summary. File not found: {pareto_csv_path}"
            )
            return

        def get_column_stats(
            df: pd.DataFrame,
            column_name: str,
        ) -> tuple[float, float, float, float] | None:
            if column_name not in df.columns:
                return None

            col_data = pd.to_numeric(df[column_name], errors="coerce").dropna()
            if len(col_data) == 0:
                return None

            min_ = float(round(col_data.min(), 4))
            max_ = float(round(col_data.max(), 4))
            avg_ = float(round(col_data.mean(), 4))
            std_ = float(round(col_data.std(ddof=1), 4)) if len(col_data) > 1 else 0.0
            return min_, max_, avg_, std_

        summary_lines = []

        # Top-level run facts
        total_generations = (
            int(generation_data["gen"].max()) + 1
            if "gen" in generation_data.columns
            else len(generation_data)
        )
        final_front_size = len(pareto_data)
        final_total_time = (
            float(generation_data["total_time"].dropna().iloc[-1])
            if "total_time" in generation_data.columns
            and not generation_data["total_time"].dropna().empty
            else None
        )

        summary_lines.append("Multi-objective Summary\n")
        summary_lines.append(f"Total generations: {total_generations}")
        summary_lines.append(f"Final Pareto front size: {final_front_size}")
        if final_total_time is not None:
            summary_lines.append(f"Total runtime: {round(final_total_time, 4)} seconds")

        # Generation-level statistics
        generation_metrics = [
            ("front_size", "Front Size"),
            ("invalids", "Invalid Individuals"),
            ("unique_inds", "Unique Individuals"),
            ("unused_search", "Unused Search (%)"),
            ("ave_genome_length", "Average Genome Length"),
            ("ave_tree_depth", "Average Tree Depth"),
            ("ave_tree_nodes", "Average Tree Nodes"),
            ("ave_used_codons", "Average Used Codons"),
            ("time_taken", "Time Per Generation"),
        ]

        headers = ["Metric", "Min", "Max", "Average(±std)"]
        summary_rows = []

        for col_name, display_name in generation_metrics:
            stats = get_column_stats(generation_data, col_name)
            if stats is None:
                continue
            summary_rows.append(
                [display_name, stats[0], stats[1], f"{stats[2]}(±{stats[3]})"]
            )

        # Final Pareto front objective stats
        for objective_name in objective_names:
            stats = get_column_stats(pareto_data, objective_name)
            if stats is None:
                continue
            summary_rows.append(
                [
                    f"Final Front: {objective_name}",
                    stats[0],
                    stats[1],
                    f"{stats[2]}(±{stats[3]})",
                ]
            )

        if summary_rows:
            summary_lines.append("\nStatistics:\n")
            summary_lines.append(
                tabulate(summary_rows, headers=headers, tablefmt="pretty")
            )

        summary = "\n".join(summary_lines)

        summary_path = f"{save_dir}/summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)

        self.logger.info(summary)
        self.logger.info(f"Summary saved to {summary_path}")

        # plots
        self.plot_multiobjective_diagnostics(objective_names)

    def save_pareto_front(
        self,
        pareto_front: list["Individual"],
        objective_names: list[str],
    ) -> None:
        save_dir = Path(get_log_dir(self.project_id))
        front_dir = save_dir / "final_pareto_front"
        front_dir.mkdir(parents=True, exist_ok=True)

        pareto_csv_path = front_dir / "front.csv"

        headers = [
            "id",
            *objective_names,
            "used_codons",
            "genome_length",
            "tree_depth",
            "phenotype_length",
            "phenotype",
        ]

        with open(pareto_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for i, ind in enumerate(pareto_front):
                if ind.fitness is None:
                    fitness = []
                elif isinstance(ind.fitness, list):
                    fitness = ind.fitness
                else:
                    fitness = [ind.fitness]

                phenotype = getattr(ind, "phenotype", None) or ""
                genotype = getattr(ind, "genotype", None)
                tree = getattr(ind, "tree", None)

                if hasattr(ind, "used_codon_count"):
                    used_codons = getattr(ind, "used_codon_count")
                elif hasattr(ind, "used_codons"):
                    used_codons = getattr(ind, "used_codons")
                else:
                    used_codons = None

                try:
                    genome_length = len(genotype) if genotype is not None else 0
                except TypeError:
                    genome_length = 0

                try:
                    tree_depth = TreeNode.from_string(tree).max_depth if tree else 0
                except Exception:
                    tree_depth = 0

                row = [
                    i,
                    *fitness,
                    used_codons,
                    genome_length,
                    tree_depth,
                    len(phenotype),
                    phenotype,
                ]
                writer.writerow(row)

        self.logger.info(f"Final Pareto front saved to {pareto_csv_path}")

        try:
            visualize_final_pareto_front(str(save_dir), objective_names)
            self.logger.info(f"Pareto front visualization saved in {save_dir}")
        except Exception as e:
            self.logger.error(f"Failed to visualize Pareto front: {e}")

    def plot_single_objective_diagnostics(self) -> None:
        save_dir = get_log_dir(self.project_id)

        try:
            plot_single_objective_fitness(save_dir)
            self.logger.info("Saved single-objective fitness plot")
        except Exception as e:
            self.logger.error(f"Failed to save fitness plot: {e}")

        try:
            plot_single_objective_complexity(save_dir)
            self.logger.info("Saved complexity plot")
        except Exception as e:
            self.logger.error(f"Failed to save complexity plot: {e}")

        try:
            plot_search_diagnostics(save_dir)
            self.logger.info("Saved search diagnostics plot")
        except Exception as e:
            self.logger.error(f"Failed to save search diagnostics plot: {e}")

        try:
            plot_generation_runtime(save_dir)
            self.logger.info("Saved runtime plot")
        except Exception as e:
            self.logger.error(f"Failed to save runtime plot: {e}")

    def plot_multiobjective_diagnostics(self, objective_names: list[str]) -> None:
        save_dir = get_log_dir(self.project_id)

        try:
            visualize_final_pareto_front(save_dir, objective_names)
            self.logger.info("Saved final Pareto front plot")
        except Exception as e:
            self.logger.error(f"Failed to save final Pareto front plot: {e}")

        try:
            visualize_pareto_front_evolution(save_dir, objective_names)
            self.logger.info("Saved Pareto front evolution plot")
        except Exception as e:
            self.logger.error(f"Failed to save Pareto front evolution plot: {e}")

        try:
            plot_front_size(save_dir)
            self.logger.info("Saved front size plot")
        except Exception as e:
            self.logger.error(f"Failed to save front size plot: {e}")

        try:
            plot_objective_progress(save_dir, objective_names)
            self.logger.info("Saved objective progress plot")
        except Exception as e:
            self.logger.error(f"Failed to save objective progress plot: {e}")

        try:
            plot_search_diagnostics(save_dir)
            self.logger.info("Saved search diagnostics plot")
        except Exception as e:
            self.logger.error(f"Failed to save search diagnostics plot: {e}")

        try:
            plot_generation_runtime(save_dir)
            self.logger.info("Saved runtime plot")
        except Exception as e:
            self.logger.error(f"Failed to save runtime plot: {e}")


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
                if not ind.fitness:
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
