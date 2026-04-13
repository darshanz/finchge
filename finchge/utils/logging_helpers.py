from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Set

import numpy as np

from finchge.core import Individual, Population
from finchge.grammar.derivation_tree import TreeNode


class PopulationMetricsHelper:
    """Computes aggregate statistics from a population."""

    @staticmethod
    def _safe_len(x: Any) -> int:
        try:
            return len(x)
        except Exception:
            return 0

    @staticmethod
    def _tree_depth(tree: Optional[str]) -> int:
        if not tree:
            return 0
        try:
            return TreeNode.from_string(tree).max_depth
        except Exception:
            return 0

    @staticmethod
    def _tree_nodes(tree: Optional[str]) -> int:
        if not tree:
            return 0
        try:
            node = TreeNode.from_string(tree)
            if hasattr(node, "node_count"):
                return int(node.node_count)
            if hasattr(node, "get_node_count"):
                return int(node.get_node_count())
            return 0
        except Exception:
            return 0

    @staticmethod
    def compute(population: Population) -> dict[str, Any]:
        individuals = population.individuals

        fitness_vals: list[float] = []
        genome_lengths: list[int] = []
        tree_depths: list[int] = []
        tree_nodes: list[int] = []
        used_codons: list[int] = []

        unique_phenotypes: set[str] = set()
        invalids = 0

        for ind in individuals:
            fitness = getattr(ind, "fitness", None)
            if fitness and len(fitness) > 0:
                value = fitness[0]
                if isinstance(value, (int, float)) and math.isfinite(value):
                    fitness_vals.append(float(value))
                else:
                    invalids += 1
            else:
                invalids += 1

            genotype = getattr(ind, "genotype", None)
            genome_lengths.append(PopulationMetricsHelper._safe_len(genotype))

            tree = getattr(ind, "tree", None)
            tree_depths.append(PopulationMetricsHelper._tree_depth(tree))
            tree_nodes.append(PopulationMetricsHelper._tree_nodes(tree))

            if hasattr(ind, "used_codon_count"):
                codons = getattr(ind, "used_codon_count", 0)
            elif hasattr(ind, "used_codons"):
                codons = getattr(ind, "used_codons", 0)
            else:
                codons = 0
            used_codons.append(int(codons) if codons is not None else 0)

            phenotype = getattr(ind, "phenotype", None)
            if phenotype:
                unique_phenotypes.add(str(phenotype))

        def stats(
            arr: Sequence[int | float],
        ) -> tuple[Optional[float], Optional[float], Optional[float]]:
            if not arr:
                return None, None, None
            return float(np.mean(arr)), float(np.max(arr)), float(np.min(arr))

        ave_genome, max_genome, min_genome = stats(genome_lengths)
        ave_depth, max_depth, min_depth = stats(tree_depths)
        ave_nodes, max_nodes, min_nodes = stats(tree_nodes)
        ave_codons, max_codons, min_codons = stats(used_codons)

        total_inds = len(individuals)
        unique_inds = len(unique_phenotypes)

        if total_inds > 0:
            unused_search = 100 - (unique_inds / total_inds * 100)
        else:
            unused_search = None

        return {
            "ave_fitness": float(np.mean(fitness_vals)) if fitness_vals else None,
            "invalids": invalids,
            "total_inds": total_inds,
            "unique_inds": unique_inds,
            "unused_search": unused_search,
            "ave_genome_length": ave_genome,
            "max_genome_length": max_genome,
            "min_genome_length": min_genome,
            "ave_tree_depth": ave_depth,
            "max_tree_depth": max_depth,
            "min_tree_depth": min_depth,
            "ave_tree_nodes": ave_nodes,
            "max_tree_nodes": max_nodes,
            "min_tree_nodes": min_nodes,
            "ave_used_codons": ave_codons,
            "max_used_codons": max_codons,
            "min_used_codons": min_codons,
        }


class LogIOHelper:
    """Small helper for filesystem and serialization operations."""

    @staticmethod
    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_json(path: Path, data: Any) -> None:
        LogIOHelper.ensure_dir(path.parent)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def write_text(path: Path, content: str) -> None:
        LogIOHelper.ensure_dir(path.parent)
        path.write_text(content)

    @staticmethod
    def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
        LogIOHelper.ensure_dir(path.parent)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    @staticmethod
    def append_csv_row(path: Path, row: list[Any]) -> None:
        LogIOHelper.ensure_dir(path.parent)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    @staticmethod
    def write_npy(path: Path, array: np.ndarray) -> None:
        LogIOHelper.ensure_dir(path.parent)
        np.save(path, array)

    @staticmethod
    def write_npz_compressed(path: Path, **arrays: Any) -> None:
        LogIOHelper.ensure_dir(path.parent)
        np.savez_compressed(path, **arrays)


class IndividualLogHelper:
    """Extracts consistent, serializable information from an individual."""

    @staticmethod
    def get_fitness(ind: Any) -> Any:
        return getattr(ind, "fitness", None)

    @staticmethod
    def get_phenotype(ind: Any) -> Optional[str]:
        return getattr(ind, "phenotype", None)

    @staticmethod
    def get_genotype(ind: Any) -> Any:
        return getattr(ind, "genotype", None)

    @staticmethod
    def get_tree(ind: Any) -> Optional[str]:
        return getattr(ind, "tree", None)

    @staticmethod
    def get_used_codons(ind: Any) -> Any:
        if hasattr(ind, "used_codon_count"):
            return getattr(ind, "used_codon_count")
        if hasattr(ind, "used_codons"):
            return getattr(ind, "used_codons")
        return None

    @staticmethod
    def get_tree_depth(tree: Optional[str]) -> int:
        if not tree:
            return 0
        try:
            return TreeNode.from_string(tree).max_depth
        except (AttributeError, ValueError, TypeError):
            return 0

    @staticmethod
    def get_phenotype_length(phenotype: Optional[str]) -> int:
        return len(phenotype) if phenotype else 0

    @staticmethod
    def get_genotype_length(genotype: Any) -> int:
        if genotype is None:
            return 0
        try:
            return len(genotype)
        except TypeError:
            return 0

    @staticmethod
    def build_summary(ind: Any) -> dict[str, Any]:
        phenotype = IndividualLogHelper.get_phenotype(ind)
        genotype = IndividualLogHelper.get_genotype(ind)
        tree = IndividualLogHelper.get_tree(ind)
        fitness = IndividualLogHelper.get_fitness(ind)

        return {
            "fitness": fitness,
            "used_codons": IndividualLogHelper.get_used_codons(ind),
            "tree_depth": IndividualLogHelper.get_tree_depth(tree),
            "phenotype_length": IndividualLogHelper.get_phenotype_length(phenotype),
            "genotype_length": IndividualLogHelper.get_genotype_length(genotype),
            "has_phenotype": phenotype is not None,
            "has_genotype": genotype is not None,
            "has_tree": tree is not None,
        }

    @staticmethod
    def write_individual_artifacts(
        base_dir: Path,
        ind: Any,
        file_stem: str,
        exclude: Set[str],
        compress_genotypes: bool,
    ) -> None:
        phenotype = IndividualLogHelper.get_phenotype(ind)
        genotype = IndividualLogHelper.get_genotype(ind)
        tree = IndividualLogHelper.get_tree(ind)

        if phenotype and "phenotypes" not in exclude:
            LogIOHelper.write_text(
                base_dir / "phenotypes" / f"{file_stem}.txt", phenotype
            )

        if "genotypes" not in exclude and genotype is not None:
            if compress_genotypes:
                LogIOHelper.write_npz_compressed(
                    base_dir / "genotypes" / f"{file_stem}.npz",
                    genotype=genotype,
                )
            else:
                LogIOHelper.write_text(
                    base_dir / "genotypes" / f"{file_stem}.txt",
                    str(genotype),
                )

        if tree and "trees" not in exclude:
            LogIOHelper.write_text(base_dir / "trees" / f"{file_stem}_tree.txt", tree)


class PopulationSamplingHelper:
    """Helper for selecting representative individuals from a population."""

    @staticmethod
    def sample_indices(population: Any, sample_size: int) -> list[int]:
        n = len(population)
        sample_size = min(sample_size, n)

        if sample_size <= 0:
            return []

        if n <= sample_size:
            return list(range(n))

        try:
            fitness_values = [ind.fitness[0] for ind in population.individuals]
            sorted_indices = np.argsort(fitness_values)

            indices: list[int] = []
            for i in range(sample_size):
                pos = int(i * (n - 1) / (sample_size - 1)) if sample_size > 1 else 0
                indices.append(int(sorted_indices[pos]))
            return indices
        except (AttributeError, IndexError, ValueError, TypeError):
            step = n / sample_size
            return [int(i * step) for i in range(sample_size)]


class ParetoFrontMetricsHelper:
    """Computes aggregate statistics for a Pareto front."""

    @staticmethod
    def compute(
        front: list[Individual],
        objective_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "front_size": len(front),
        }

        if not front:
            return stats

        try:
            fitness_matrix = np.array([ind.fitness for ind in front], dtype=float)
        except Exception:
            return stats

        if fitness_matrix.ndim != 2 or fitness_matrix.shape[0] == 0:
            return stats

        n_objectives = fitness_matrix.shape[1]
        names = objective_names or [f"obj{i}" for i in range(n_objectives)]

        for i, name in enumerate(names):
            col = fitness_matrix[:, i]
            finite_col = col[np.isfinite(col)]

            if len(finite_col) == 0:
                stats[f"front_min_{name}"] = None
                stats[f"front_max_{name}"] = None
                stats[f"front_mean_{name}"] = None
            else:
                stats[f"front_min_{name}"] = float(np.min(finite_col))
                stats[f"front_max_{name}"] = float(np.max(finite_col))
                stats[f"front_mean_{name}"] = float(np.mean(finite_col))

        return stats
