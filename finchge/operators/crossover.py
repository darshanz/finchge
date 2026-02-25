import logging
from typing import List, Optional, Tuple

from finchge.core.individual import Individual
from finchge.grammar.derivation_tree import TreeNode
from finchge.operators.base import GECrossoverStrategy


class OnePointCrossover(GECrossoverStrategy):
    """
    One-point crossover operator.

    A single crossover point is selected in each parent
    genome, and the genome segments after this point are exchanged to form
    offspring.  Parent genomes are split at a randomly selected point. Offspring are
    constructed by concatenating prefix segments from one parent with suffix
    segments from the other.

    """

    def __init__(
        self,
        codon_size: int,
        crossover_proba: float,
        within_used: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the one-point crossover strategy.

        Args:
            codon_size: Size of each codon.
            crossover_proba: Probability of crossover occurring.
            within_used: If True, crossover points are limited to the
                used portion of each parent's genome.
        """
        super().__init__(crossover_proba=crossover_proba, random_state=random_state)
        self.codon_size = codon_size
        self.within_used = within_used

    def cross(
        self,
        parent1: "Individual",
        parent2: "Individual",
    ) -> Tuple["Individual", "Individual"]:
        """
        Perform one-point crossover

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            Two offspring individuals.
        """
        if parent1.genotype is None or parent2.genotype is None:
            raise ValueError("The genotype must exist.")

        genome_p1: List[int] = parent1.genotype.copy()
        genome_p2: List[int] = parent2.genotype.copy()

        len_p1 = len(genome_p1)
        len_p2 = len(genome_p2)

        used_0 = parent1.used_codon_count or 0
        used_1 = parent2.used_codon_count or 0

        # Determine maximum valid crossover points
        if self.within_used:
            max_p_0 = max(1, min(used_0, len_p1))
            max_p_1 = max(1, min(used_1, len_p2))
        else:
            max_p_0 = max(1, len_p1)
            max_p_1 = max(1, len_p2)

        if self.rng.random() < self.crossover_proba and max_p_0 > 1 and max_p_1 > 1:
            try:
                # Select crossover points
                pt_p_0 = self.rng.randint(1, max_p_0)
                pt_p_1 = self.rng.randint(1, max_p_1)

                # Create offspring by swapping genome tails
                genome_o1: List[int] = genome_p1[:pt_p_0] + genome_p2[pt_p_1:]
                genome_o2: List[int] = genome_p2[:pt_p_1] + genome_p1[pt_p_0:]
            except (IndexError, ValueError) as exc:
                # return parent copies if error
                logging.debug("Crossover failed (%s); using parent copies.", exc)
                genome_o1 = genome_p1.copy()
                genome_o2 = genome_p2.copy()
        else:
            # when no crossover return parent copies
            genome_o1 = genome_p1.copy()
            genome_o2 = genome_p2.copy()

        return (
            Individual.from_genotype(genome_o1),
            Individual.from_genotype(genome_o2),
        )


class TwoPointCrossover(GECrossoverStrategy):
    """
    Two-point crossover generalizes one-point crossover by selecting two
    crossover points in each parent genome and exchanging the intermediate
    segments.

    Offspring are formed by combining prefix and suffix regions from one
    parent with the middle region from the other parent.

    """

    def __init__(
        self,
        codon_size: int,
        crossover_proba: float,
        within_used: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the two-point crossover .

        Args:
            codon_size: Size of each codon.
            crossover_proba: Probability of crossover occurring.
            within_used: If True, crossover points are limited to the
                used portion of each parent's genome.
        """
        super().__init__(crossover_proba=crossover_proba, random_state=random_state)
        self.codon_size = codon_size
        self.within_used = within_used

    def cross(
        self,
        parent1: "Individual",
        parent2: "Individual",
    ) -> Tuple["Individual", "Individual"]:
        """Perform two-point crossover.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            Two offspring individuals.
        """
        if parent1.genotype is None or parent2.genotype is None:
            raise ValueError("The genotype must exist.")

        genome_p1: List[int] = parent1.genotype.copy()
        genome_p2: List[int] = parent2.genotype.copy()

        len_p1 = len(genome_p1)
        len_p2 = len(genome_p2)

        used_0 = parent1.used_codon_count or 0
        used_1 = parent2.used_codon_count or 0

        # Determine maximum valid crossover range
        if self.within_used:
            max_p_0 = max(0, min(used_0, len_p1))
            max_p_1 = max(0, min(used_1, len_p2))
        else:
            max_p_0 = len_p1
            max_p_1 = len_p2

        # Two-point crossover requires at least two valid cut points
        if self.rng.random() < self.crossover_proba and max_p_0 >= 2 and max_p_1 >= 2:
            # Select two ordered crossover points per parent
            pt_p_0_1 = self.rng.randint(1, max_p_0 - 1)
            pt_p_0_2 = self.rng.randint(pt_p_0_1 + 1, max_p_0)

            pt_p_1_1 = self.rng.randint(1, max_p_1 - 1)
            pt_p_1_2 = self.rng.randint(pt_p_1_1 + 1, max_p_1)

            # Exchange middle segments
            genome_o1: List[int] = (
                genome_p1[:pt_p_0_1]
                + genome_p2[pt_p_1_1:pt_p_1_2]
                + genome_p1[pt_p_0_2:]
            )
            genome_o2: List[int] = (
                genome_p2[:pt_p_1_1]
                + genome_p1[pt_p_0_1:pt_p_0_2]
                + genome_p2[pt_p_1_2:]
            )
        else:
            # No crossover: return parent copies
            genome_o1 = genome_p1.copy()
            genome_o2 = genome_p2.copy()

        return (
            Individual.from_genotype(genome_o1),
            Individual.from_genotype(genome_o2),
        )


class UniformCrossover(GECrossoverStrategy):
    """
    Uniform crossover operator.

    Uniform crossover recombines parent genomes by independently swapping genes
    between parents at each position with fixed probability.

    Instead of using fixed crossover points, uniform crossover applies a
    binary mixing mask across the genome, providing maximal mixing of
    parental genetic material.
    """

    def __init__(
        self,
        crossover_proba: float,
        within_used: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize uniform crossover

        Args:
            crossover_proba: Probability of crossover occurring.
            within_used: If True, crossover is limited to the
                used portion of each parent's genome.
        """
        super().__init__(crossover_proba=crossover_proba, random_state=random_state)
        self.within_used = within_used

    def cross(
        self,
        parent1: "Individual",
        parent2: "Individual",
    ) -> Tuple["Individual", "Individual"]:
        """Perform uniform crossover.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            Two offspring individuals.
        """
        if parent1.genotype is None or parent2.genotype is None:
            raise ValueError("The genotype must exist.")

        genome_p1: List[int] = parent1.genotype.copy()
        genome_p2: List[int] = parent2.genotype.copy()

        len_p1 = len(genome_p1)
        len_p2 = len(genome_p2)

        used_0 = parent1.used_codon_count or 0
        used_1 = parent2.used_codon_count or 0

        # Determine crossover range
        if self.within_used:
            max_p_0 = max(0, min(used_0, len_p1))
            max_p_1 = max(0, min(used_1, len_p2))
        else:
            max_p_0 = len_p1
            max_p_1 = len_p2

        if self.rng.random() < self.crossover_proba:
            genome_o1: List[int] = genome_p1.copy()
            genome_o2: List[int] = genome_p2.copy()

            # Only swap within the overlapping crossover range
            min_length = min(max_p_0, max_p_1)

            for i in range(min_length):
                if self.rng.random() < 0.5:
                    genome_o1[i], genome_o2[i] = genome_o2[i], genome_o1[i]
        else:
            # No crossover: return parent copies
            genome_o1 = genome_p1.copy()
            genome_o2 = genome_p2.copy()

        return (
            Individual.from_genotype(genome_o1),
            Individual.from_genotype(genome_o2),
        )


class SubtreeCrossover(GECrossoverStrategy):
    """
    Subtree crossover exchanges randomly selected subtrees between two parent
    program trees while preserving syntactic correctness with respect to a
    grammar.

    Two compatible non-terminal nodes are selected from parent trees, and
    the subtrees rooted at these nodes are swapped.


    """

    def __init__(
        self,
        crossover_proba: float,
        non_terminals: list[str],
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the subtree crossove.

        Args:
            crossover_proba: Probability of crossover occurring.
            non_terminals: List of grammar non-terminal symbols that are valid
                crossover points.
        """
        super().__init__(crossover_proba=crossover_proba, random_state=random_state)
        self.non_terminals: list[str] = non_terminals

    def cross(
        self, p_0: "Individual", p_1: "Individual"
    ) -> Tuple["Individual", "Individual"]:
        """
        Perform subtree crossover between two parent individuals.

        If crossover does not occur, cloned trees are returned as new
        individuals.

        Args:
            **kwargs:
            p_0: First parent individual.
            p_1: Second parent individual.

        Returns:
            A tuple containing two newly constructed offspring individuals.
        """
        if p_0.tree is None or p_1.tree is None:
            raise ValueError("The tree must exist.")

        # Clone trees to ensure parents remain untouched
        t0: TreeNode = TreeNode.from_string(p_0.tree)
        t1: TreeNode = TreeNode.from_string(p_1.tree)

        # Decide whether crossover occurs
        if self.rng.random() > self.crossover_proba:
            # No crossover: return fresh individuals built from cloned trees
            return (
                Individual.from_tree(t0),
                Individual.from_tree(t1),
            )

        # Identify shared non-terminal symbols as valid crossover points
        symbols1_set = set(t1.collect_symbols(self.non_terminals))
        shared_symbols = sorted(
            [
                symbol
                for symbol in t0.collect_symbols(self.non_terminals)
                if symbol in symbols1_set
            ]
        )

        # if there are no valid crossover points, we can just swap whole trees
        # Basically returning same parents
        if not shared_symbols:
            return (
                Individual.from_tree(t1),
                Individual.from_tree(t0),
            )

        # Select crossover point
        symbol: str = self.rng.choice(shared_symbols)

        # Select concrete nodes with the chosen non-terminal symbol
        n0: TreeNode = self.rng.choice(t0.find_by_symbol(symbol))
        n1: TreeNode = self.rng.choice(t1.find_by_symbol(symbol))

        # Perform subtree swap
        new_t0, new_t1 = n0.swap_subtree_with(n1)

        return (
            Individual.from_tree(new_t0),
            Individual.from_tree(new_t1),
        )
