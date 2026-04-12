from typing import Any, Dict, Optional, Type, TypeVar

import numpy as np
from tabulate import tabulate

from finchge.grammar.derivation_tree import TreeNode
from finchge.operators.base import GEMutationStrategy

T = TypeVar("T")


class Individual:
    """
    Represents a single individual in an evolutionary algorithm.

    An Individual encapsulates the genetic representation (genotype), its expressed
    form (phenotype), and metadata produced during evaluation and selection.
    Fitness values are not assigned at construction time and must be computed by
    a [FitnessEvaluator][finchge.fitness.FitnessEvaluator].

    Algorithm-specific information (e.g., Pareto rank, crowding distance, or
    reference-point association) is stored in the `meta` dictionary and is expected
    to be populated and cleared by the evolutionary algorithm.

    Attributes:
        genotype (Optional[list[int]]):
            Genetic representation of the individual. A list of integers or None
            if not yet initialized or derived from a tree-based initialiser.

        phenotype (str):
            Phenotypic representation derived from the genotype via grammar-based
            mapping. Empty until mapping occurs.

        used_genome (Optional[list[int]]):
            Portion of the genotype consumed during genotype-to-phenotype mapping.
            Set during evaluation.

        used_codon_count (Optional[int]):
            Number of codons consumed during mapping.

        invalid (bool):
            Indicates whether the individual is invalid (e.g., mapping failure).
            Invalid individuals should not participate in fitness-based selection.

        tree (Optional[str]):
            Serialized derivation tree (e.g., JSON) produced during mapping or
            tree-based initialisation.

        fitness (list[float]):
            Fitness values assigned during evaluation. Empty until evaluated.
            Supports both single-objective and multi-objective optimization.

        meta (Dict[str, Any]):
            Algorithm-specific metadata managed by the evolutionary algorithm,
            such as Pareto rank, crowding distance, or dominance information.
    """

    CASE_DATA_META_KEY = "_selection_case_data"

    def __init__(
        self,
        *,
        genotype: Optional[list[int]] = None,
        phenotype: str | None = None,
        used_genome: Optional[list[int]] = None,
        used_codon_count: int = 0,
        invalid: bool = False,
        tree: Optional[str] = None,
    ) -> None:
        """
        Initialize an Individual.

        Args:
            genotype (Optional[list[int]]):
                Genetic representation as a list of integers, or None if the
                individual is initialized from a tree or will be derived later.

            phenotype (str):
                Phenotypic representation of the individual. Defaults to an empty
                string and is populated during mapping.

            used_genome (Optional[list[int]]):
                Portion of the genotype actually consumed during mapping.
                Populated during evaluation.

            used_codon_count (Optional[int]):
                Number of codons consumed during genotype-to-phenotype mapping.

            invalid (bool):
                Whether the individual is invalid (e.g., mapping failure).

            tree (Optional[str]):
                Optional serialized derivation tree (e.g., JSON) associated with
                this individual.
        """
        # genotype should be of type list of integers : if present
        if genotype is not None:
            if not isinstance(genotype, list):
                raise TypeError("genotype must be a list[int] or None")
            if not all(isinstance(g, int) for g in genotype):
                raise TypeError("genotype must contain only integers")

        # Similarly used genotype is also of list of integers : if present (exist only after mapping)
        if used_genome is not None:
            if not isinstance(used_genome, list):
                raise TypeError("used_genotype must be a list[int] or None")
            if not all(isinstance(g, int) for g in used_genome):
                raise TypeError("used_genotype must contain only integers")

        self.genotype: Optional[list[int]] = genotype
        self.phenotype: str | None = phenotype
        self.used_genome: Optional[list[int]] = used_genome
        self.used_codon_count: int = used_codon_count
        self.invalid: bool = invalid
        self.tree: Optional[str] = tree

        # Fitness is empty list until the individual is evaluated.
        # List is used to support multiple objective optimization
        self.fitness: list[float] = []

        # algorithm-specific metadata will be in self.meta if required by any problem
        self.meta: Dict[str, Any] = {}

    # Class methods for easy creation of individuals by initialisers
    @classmethod
    def from_genotype(cls, genotype: list[int]) -> "Individual":
        """
        Construct an Individual from a genotype.

        Args:
            genotype (list[int]): Genetic representation as a list of integers.

        Returns: A new Individual initialized with the given genotype.
        """
        return cls(genotype=genotype)

    @classmethod
    def from_tree(cls, tree: TreeNode) -> "Individual":
        """
        Construct an Individual from a derivation tree.

        Args:
            tree (str):
                Root of the derivation tree used to initialize the individual.

        Returns:
            A new Individual initialized with the given tree representation.
        """
        return cls(tree=tree.to_string())

    def mutated(self, mutation_strategy: GEMutationStrategy) -> "Individual":
        """
        Apply a mutation strategy to this individual.

        Args:
            mutation_strategy (GEMutationStrategy):
                Mutation operator used to produce a mutated individual.

        Returns:
            A new Individual representing the mutated offspring.
        """
        return mutation_strategy.mutate(self)

    def clone(self) -> "Individual":
        """
        Create a deep copy of this individual.

        Returns:
            A new Individual with copied genotype, phenotype, fitness, and metadata.
        """
        clone = Individual(
            genotype=self.genotype.copy() if self.genotype is not None else None,
            phenotype=self.phenotype,
            used_genome=(
                self.used_genome.copy() if self.used_genome is not None else None
            ),
            used_codon_count=self.used_codon_count,
            invalid=self.invalid,
            tree=self.tree,
        )
        clone.fitness = self.fitness.copy()
        clone.meta = self.meta.copy()
        return clone

    def get_meta(self, key: str, expected_type: Type[T] | None = None) -> Any:
        """
        Retrieve algorithm-specific metadata stored on the individual.

        Args:
            key:
                Metadata key to retrieve.

            expected_type:
                Expected type of the metadata value if being strict.

        Raises:
            ValueError:
                If the key is missing or the value does not match the expected type.

        Returns:
            The metadata value associated with the given key.
        """
        if key not in self.meta:
            raise ValueError(f"Missing meta key '{key}'")

        value = self.meta.get(key)
        if expected_type is not None and not isinstance(value, expected_type):
            raise ValueError(
                f"Individual.meta['{key}'] must be of type {expected_type.__name__}"
            )
        return value

    def set_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    def has_meta(self, key: str) -> bool:
        return key in self.meta

    def remove_meta(self, key: str) -> None:
        self.meta.pop(key, None)

    def has_fitness(self) -> bool:
        return bool(self.fitness)

    def has_finite_fitness(self) -> bool:
        return self.has_fitness() and all(np.isfinite(v) for v in self.fitness)

    def has_case_data(self, key: str) -> bool:
        case_store = self.meta.get(self.CASE_DATA_META_KEY)
        return isinstance(case_store, dict) and key in case_store

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the individual.

        This method is intended for end-user logging and quick inspection in
        scripts and notebooks.

        Returns:
            str: A readable summary containing phenotype, fitness, and basic flags.
        """
        fitness_str = "Not evaluated" if not self.fitness else str(self.fitness)
        geno_len = len(self.genotype) if self.genotype is not None else 0
        return (
            "Individual(\n"
            f"  phenotype={self.phenotype!r},\n"
            f"  fitness={fitness_str},\n"
            f"  invalid={self.invalid},\n"
            f"  genotype_len={geno_len}\n"
            ")"
        )

    def __repr__(self) -> str:
        """
        Return a developer-facing representation of the individual.

        This method is intended for debugging and interactive sessions.

        Returns:
            str: A concise representation including phenotype and fitness status.
        """
        fitness_str = "Not evaluated" if not self.fitness else str(self.fitness)
        return f"Individual(phenotype={self.phenotype!r}, fitness={fitness_str})"

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the individual for Jupyter notebooks.

        Jupyter calls this method automatically to render rich HTML output.
        If `tabulate` is available, a neat table is rendered. Otherwise, a small
        HTML fallback is returned.

        Returns:
            str: HTML representation of the individual.
        """
        from finchge.utils.display_utils import is_jupyter

        in_notebook: bool = is_jupyter()

        genotype_display: Any
        if self.genotype is None:
            genotype_display = "Not set"
        else:
            # Keep display short for readability
            max_show = 30
            if len(self.genotype) > max_show:
                genotype_display = self.genotype[:max_show] + ["..."]
            else:
                genotype_display = self.genotype

        fitness_display: Any = "Not evaluated" if not self.fitness else self.fitness
        used_genotype_display: Any = (
            "Not set" if self.used_genome is None else self.used_genome
        )
        tree_display: Any = "Not set" if self.tree is None else "Available"

        rows = [
            ["Phenotype", self.phenotype if self.phenotype else "Not mapped"],
            ["Genotype", genotype_display],
            ["Used Genotype", used_genotype_display],
            [
                "Used Codons",
                (
                    self.used_codon_count
                    if self.used_codon_count is not None
                    else "Not set"
                ),
            ],
            ["Invalid", self.invalid],
            ["Fitness", fitness_display],
            ["Tree", tree_display],
        ]

        tablefmt = "html" if in_notebook else "simple"
        return tabulate(rows, headers=["Attribute", "Value"], tablefmt=tablefmt)
