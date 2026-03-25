from typing import Any

from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode


class TreeGenerator:
    """
    Responsible for generating grammar-valid derivation trees.

    This class is used by:
    - population initialisers
    - subtree mutation
    """

    def __init__(self, grammar: "Grammar", max_tree_depth: int) -> None:
        self.grammar = grammar
        self.max_tree_depth = max_tree_depth
        # Analyze grammar ONCE
        self.grammar.analyze()

    def generate_tree(
        self,
        *,
        max_depth: int,
        rng: Any,
        start_symbol: str | None = None,
        force_full: bool = False,
        position_independent: bool = False,
        strict_full: bool = True,  # fail if grammar is not compatible if false , just fall back
    ) -> "TreeNode":
        """
        Generate a grammar-valid derivation tree.

        This method constructs a derivation tree by recursively expanding
        non-terminal symbols using productions defined in the grammar. The tree
        is generated without a genotype (genome-free construction) and is primarily
        used for population initialisation and subtree mutation.

        Tree construction supports both Grow-style and Full-style initialisation,
        as well as position-independent expansion.

        Args:
            max_depth:
                Maximum allowed tree depth. Expansion of non-terminals stops when
                this depth is reached.

            rng:
                Random number generator used for stochastic production selection.

            start_symbol:
                Non-terminal symbol used as the root of the tree. If None,
                the grammar start rule is used.

            force_full:
                If True, enforces Full-style initialisation where internal nodes
                must expand using productions containing at least one non-terminal
                until the maximum depth is reached. If False, Grow-style expansion
                is used.

            position_independent:
                If True, uses position-independent expansion (PI-Grow), where nodes
                are expanded in random order rather than strict depth-first order.

            strict_full:
                Controls behaviour when Full initialisation is requested but the
                grammar does not allow a strictly full derivation.

                - If True:
                    Enforces strict Koza-style Full initialisation. Tree generation
                    raises a RuntimeError if valid Full productions cannot be found.

                - If False reminder:
                    Falls back to Grow-style production selection when Full expansion
                    becomes impossible.

        Returns:
            TreeNode:
                Root node of the generated derivation tree.

        Raises:
            RuntimeError:
                If strict_full is True and the grammar constraints prevent Full
                initialisation, or if no valid productions exist under the given
                constraints.

        References:
            Koza, J. R. (1992).
            Genetic Programming: On the Programming of Computers by Means of
            Natural Selection. MIT Press.

            O’Neill, M., & Ryan, C. (2003).
            Grammatical Evolution: Evolutionary Automatic Programming in a
            Arbitrary Language. Springer.

            Luke, S. (2000).
            Two Fast Tree-Creation Algorithms for Genetic Programming.
            IEEE Transactions on Evolutionary Computation.
        """
        if not start_symbol:
            start_symbol = self.grammar.start_rule

        # The smaller value is taken for effective tree depth.
        # If global max_depth is set to smaller value, the initializers' max depth may not be effective.
        # Config Validator should validate the params.
        effective_depth = min(max_depth, self.max_tree_depth)
        return self._generate_derivation_tree(
            max_depth=effective_depth,
            start_symbol=start_symbol,
            force_full=force_full,
            position_independent=position_independent,
            rng=rng,
            strict_full=strict_full,
        )

    def generate_subtree(self, *, symbol: str, max_depth: int, rng: Any) -> "TreeNode":
        """
        Generate a subtree rooted at the given non-terminal symbol.
        Used by mutation.
        """
        if max_depth is None:
            effective_depth = self.max_tree_depth
        else:
            effective_depth = min(max_depth, self.max_tree_depth)
        return self._generate_derivation_tree(
            start_symbol=symbol, max_depth=effective_depth, rng=rng
        )

    def _generate_derivation_tree(
        self,
        max_depth: int,
        start_symbol: str,  # we may need to start in the middle
        rng: Any,
        force_full: bool = False,
        position_independent: bool = False,
        max_tries: int = 50,
        strict_full: bool = True,
    ) -> TreeNode:
        """
        Generate a derivation tree directly from the grammar.
        This is genome-free tree construction.
        """

        # Should not use original grammar's start rule(self.start_rule),
        # use the value passed (start_symbol).
        # Because, this method is also used to generate subtrees starting from given node.
        start_rule = self.grammar.rules[start_symbol]

        # Error for impossible Full mode
        if force_full and strict_full and start_rule.min_path is not None:
            if start_rule.min_path > max_depth:
                raise RuntimeError(
                    f"Full initialisation impossible: minimum derivation depth "
                    f"({start_rule.min_path}) exceeds max_depth ({max_depth})."
                )

        last_error: Exception | None = None

        for _ in range(max_tries):
            root = TreeNode(start_symbol, depth=0)
            try:
                if position_independent:
                    self._expand_pi(
                        root=root,
                        max_depth=max_depth,
                        force_full=force_full,
                        strict_full=strict_full,
                        rng=rng,
                    )
                else:
                    self._expand_depthfirst(
                        node=root,
                        max_depth=max_depth,
                        force_full=force_full,
                        strict_full=strict_full,
                        rng=rng,
                    )
                return root
            except RuntimeError as e:
                last_error = e
                continue

        # Build a helpful message
        mode = "Full" if force_full else "Grow"
        msg = (
            f"Failed to generate derivation tree using {mode} initialisation.\n"
            f"max_depth={max_depth}, "
            f"min_path={start_rule.min_path}, "
            f"max_path={'∞' if start_rule.max_path is None else start_rule.max_path}.\n"
            "Consider using PI-Grow or adjusting grammar depth constraints."
        )

        if force_full:
            msg += (
                "\n\nRecommendation:\n"
                "- Use PI-Grow initialisation or\n"
                "- Disable Full mode in Ramped Half-and-Half, or\n"
            )

        raise RuntimeError(msg) from last_error

    def _expand_depthfirst(
        self,
        node: TreeNode,
        max_depth: int,
        force_full: bool,
        strict_full: bool,
        rng: Any,
    ) -> None:
        # Terminals are already fully expanded
        if node.symbol not in self.grammar.non_terminals:
            return

        rule = self.grammar.rules[node.symbol]
        productions_all = rule.choices

        # Remaining depth budget for children
        remaining = max_depth - (node.depth + 1)

        def is_terminal_only(prod: list[str]) -> bool:
            return all(sym not in self.grammar.non_terminals for sym in prod)

        def has_any_nonterminal(prod: list[str]) -> bool:
            return any(sym in self.grammar.non_terminals for sym in prod)

        def is_depth_feasible(prod: list[str]) -> bool:
            """
            Production is feasible if all non-terminal children
            can terminate within remaining depth.
            """
            for sym in prod:
                if sym in self.grammar.non_terminals:
                    child_rule = self.grammar.rules[sym]

                    # Child cannot terminate at all
                    if child_rule.min_path is None:
                        return False

                    # Child requires deeper tree than remaining depth
                    if child_rule.min_path > remaining:
                        return False

            return True

        # Build candidate productions under current constraints
        if node.depth >= max_depth:
            # Depth limit reached :: must terminate immediately
            candidates = [p for p in productions_all if is_terminal_only(p)]

        elif force_full:
            # Full initialisation:
            # Must grow AND must remain depth-feasible
            candidates = [
                p
                for p in productions_all
                if has_any_nonterminal(p) and is_depth_feasible(p)
            ]

            if not candidates:
                if not strict_full:
                    # fallback :: behave like Grow but remain feasible
                    candidates = [p for p in productions_all if is_depth_feasible(p)]

                    # If still nothing feasible, last resort = terminal productions
                    if not candidates:
                        candidates = [p for p in productions_all if is_terminal_only(p)]

                if not candidates:
                    raise RuntimeError(
                        f"Full initialisation impossible for symbol={node.symbol!r} "
                        f"at depth={node.depth} with max_depth={max_depth} "
                        f"(no depth-feasible productions)."
                    )

        else:
            # Grow initialisation:
            # Grow :: Ryan Sensible Initialisation behaviour
            # Reference : Section 3.2 (https://link.springer.com/chapter/10.1007/3-540-36599-0_37)
            #  "In case of grow, recursive and non-recursive rules are chosen with equal probability."

            feasible = [p for p in productions_all if is_depth_feasible(p)]

            recursive = [p for p in feasible if has_any_nonterminal(p)]
            terminating = [p for p in feasible if is_terminal_only(p)]

            if recursive and terminating:
                # Ryan et al..suggest 50/50 group selection
                # (https://link.springer.com/chapter/10.1007/3-540-36599-0_37)
                if rng.random() < 0.5:
                    candidates = recursive
                else:
                    candidates = terminating
            elif recursive:
                candidates = recursive
            elif terminating:
                candidates = terminating
            else:
                raise RuntimeError(
                    f"No valid productions for symbol={node.symbol!r} "
                    f"at depth={node.depth} under max_depth={max_depth}."
                )

        # Select production
        production = rng.choice(candidates)

        # Expand children in production order
        for sym in production:
            child = TreeNode(sym)
            node.add_child(child)

            self._expand_depthfirst(
                node=child,
                max_depth=max_depth,
                force_full=force_full,
                strict_full=strict_full,
                rng=rng,
            )

    def _expand_pi(
        self,
        root: TreeNode,
        max_depth: int,
        force_full: bool,
        strict_full: bool,
        rng: Any,
    ) -> None:
        frontier = [root]
        forced = rng.choice(frontier)

        while frontier:
            node = frontier.pop(rng.randrange(len(frontier)))

            # Terminals are done
            if node.symbol not in self.grammar.non_terminals:
                continue

            rule = self.grammar.rules[node.symbol]
            productions_all = rule.choices
            productions = productions_all

            # Depth constraint: must terminate
            if node.depth >= max_depth:
                productions = [
                    p
                    for p in productions_all
                    if all(sym not in self.grammar.non_terminals for sym in p)
                ]

            # Full or forced expansion: must grow
            elif force_full or node is forced:
                productions = [
                    p
                    for p in productions_all
                    if any(sym in self.grammar.non_terminals for sym in p)
                ]

            # A non-terminal with no valid productions is an INVALID TREE
            if not productions:
                if (
                    (force_full or node is forced)
                    and not strict_full
                    and node.depth < max_depth
                ):
                    productions = productions_all
                else:
                    raise RuntimeError(
                        f"No valid productions for symbol '{node.symbol}' at depth {node.depth} "
                        f"(max_depth={max_depth}, force_full={force_full}, strict_full={strict_full})"
                    )

            production = rng.choice(productions)

            for sym in production:
                child = TreeNode(sym)
                node.add_child(child)
                frontier.append(child)

    def generate_tree_ptc2(
        self,
        target_size: int,
        rng: Any,
        start_symbol: str | None = None,
        max_depth: int | None = None,
    ) -> TreeNode:
        """
        Generates a derivation tree using the Probabilistic Tree-creation 2 (PTC2) algorithm.

        This implementation follows the 'refined' version of PTC2 recommended by Nicolau (2017).
        It treats the target size as the number of non-terminal expansions. By default, it
        operates without a depth limit to avoid structural bias, but it can be constrained
        to behave as the PTC2D variant by providing a `max_depth`.

        Note:
            "The best results were obtained by a refined version of the PTC2 algorithm...
            as it sampled a wider variety of tree shapes and solution lengths." (Nicolau, 2017).

        Args:
            target_size: The target number of non-terminal expansions to perform.
            rng: A random number generator (e.g., random.Random).
            start_symbol: The grammar symbol to start from. Defaults to grammar's start rule.
            max_depth: Optional maximum derivation tree depth. If None, performs
                refined PTC2. If set, performs PTC2D.

        Returns:
            TreeNode: The root of the generated derivation tree.

        Raises:
            ValueError: If target_size is not a positive integer.
        """
        grammar = self.grammar
        if target_size <= 0:
            raise ValueError("Target size must be a positive integer.")

        if start_symbol is None:
            start_symbol = grammar.start_rule

        # Initialize the tree with the root symbol
        root = TreeNode(start_symbol, depth=0)

        # The 'frontier' stores all currently active non-terminals in the derivation tree.
        frontier: list[TreeNode] = [root]
        expansions_done = 0

        # Helper function to get the minimum expansions required to terminate a symbol.
        # PTC2 uses a pre-calculated table of min-depth to ensure validity. (Ref: Section 3.4, Nicolau, 2017)
        def get_min_exp(sym: str) -> int:
            rule = grammar.rules.get(sym)
            return rule.min_path if (rule and rule.min_path is not None) else 0

        while frontier:
            # Random Node Selection.

            # "In the case of PTC2, the next non-terminal to be expanded is chosen
            # randomly from the list of all currently open non-terminals." (Nicolau, 2017).
            # This is important to avoid the 'leftmost' bias found in standard GE mapping.
            node = frontier.pop(rng.randrange(len(frontier)))

            if node.symbol not in grammar.non_terminals:
                continue

            rule = grammar.rules[node.symbol]

            # RE = Remaining Expansions budget.
            RE = target_size - expansions_done

            # f_cost = The minimum number of expansions needed to close the rest of the tree.
            f_cost = sum(get_min_exp(n.symbol) for n in frontier)

            # Filter productions based on feasibility.
            feasible = []
            for prod in rule.choices:
                # PTC2D Logic: Enforce a strict maximum depth if provided.
                #  "depth-limited versions (PTC2D) often create "bushy" trees
                #  that perform worse than the refined version." - (Nicolau, 2017).
                if max_depth is not None:
                    if node.depth + 1 > max_depth:
                        continue
                    # Depth-Feasibility
                    # Can all children terminate before max_depth?
                    if any(
                        get_min_exp(s) + node.depth + 1 > max_depth
                        for s in prod
                        if s in grammar.non_terminals
                    ):
                        continue

                # Size-Feasibility
                # Can we expand this and still terminate all other branches?
                # Cost = 1 (current expansion) + minimum expansions needed for children.
                prod_cost = 1 + sum(
                    get_min_exp(s) for s in prod if s in grammar.non_terminals
                )

                if prod_cost + f_cost <= RE:
                    feasible.append(prod)

            # Select the production.
            if not feasible:
                # Fallback: If budget is exhausted, choose the production with minimum growth.
                # "If no production is feasible... the one with the smallest
                # required number of expansions is chosen." (Nicolau, 2017).
                production = min(
                    rule.choices,
                    key=lambda p: sum(
                        get_min_exp(s) for s in p if s in grammar.non_terminals
                    ),
                )
            else:
                # Probabilistic Choice: Bias toward recursion if expansion budget remains.
                # "PTC2 allows a user to specify a requested size... and distributes
                # that size across a variety of tree shapes." (Nicolau, 2017).
                recursive = [
                    p for p in feasible if any(s in grammar.non_terminals for s in p)
                ]

                # Use the 50/50 group-selection probability recommended by Luke (2000) and Nicolau.
                if RE > 1 and recursive and rng.random() < 0.5:
                    production = rng.choice(recursive)
                else:
                    production = rng.choice(feasible)

            # Step 4: Update tree structure and frontier.
            for sym in production:
                child = TreeNode(sym, depth=node.depth + 1)
                node.add_child(child)
                if sym in grammar.non_terminals:
                    frontier.append(child)

            # Increment global expansion counter
            expansions_done += 1

        return root
