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
        # Keep grammar metadata ready for the generation methods below.
        self.grammar.analyze()

    def generate_tree_grow(
        self,
        *,
        max_depth: int,
        rng: Any,
        start_symbol: str | None = None,
    ) -> "TreeNode":
        if not start_symbol:
            start_symbol = self.grammar.start_rule

        effective_depth = min(max_depth, self.max_tree_depth)
        return self._generate_depthfirst_tree(
            start_symbol=start_symbol,
            max_depth=effective_depth,
            rng=rng,
            method="grow",
        )

    def generate_tree_full(
        self,
        *,
        max_depth: int,
        rng: Any,
        start_symbol: str | None = None,
        strict: bool = True,
    ) -> "TreeNode":
        if not start_symbol:
            start_symbol = self.grammar.start_rule

        effective_depth = min(max_depth, self.max_tree_depth)
        return self._generate_depthfirst_tree(
            start_symbol=start_symbol,
            max_depth=effective_depth,
            rng=rng,
            method="full",
            strict=strict,
        )

    def generate_tree_pi_grow(
        self,
        *,
        max_depth: int,
        rng: Any,
        start_symbol: str | None = None,
    ) -> "TreeNode":
        """
        Generate a derivation tree using PI-Grow.

        PI-Grow expands non-terminals in random queue order and keeps forcing
        growth until at least one branch reaches the requested depth.

        PI-Grow is implemented separately from generic tree generation.
        Although it is also "position independent", the literature defines a
        specific growth policy that differs from ordinary random frontier expansion.

        Reference: PonyGE2.
        """
        if not start_symbol:
            start_symbol = self.grammar.start_rule

        effective_depth = min(max_depth, self.max_tree_depth)
        return self._generate_pi_grow_tree(
            start_symbol=start_symbol,
            max_depth=effective_depth,
            rng=rng,
        )

    def _generate_pi_grow_tree(
        self,
        *,
        start_symbol: str,
        max_depth: int,
        rng: Any,
    ) -> TreeNode:
        """
        Generate a tree using PI-Grow.

        PI-Grow expands non-terminals in random queue order and keeps forcing
        recursive growth until at least one branch reaches the requested depth.
        After that, it relaxes to grow-style expansion.
        """
        root = TreeNode(start_symbol)

        # Queue holds currently expandable non-terminals.
        queue: list[TreeNode] = [root]

        while queue:
            chosen_index = rng.randrange(len(queue))
            node = queue.pop(chosen_index)

            if node.symbol not in self.grammar.non_terminals:
                continue

            overall_depth = root.max_depth
            recursive_in_queue = any(
                self._is_recursive_symbol(item.symbol) for item in queue
            )

            legal = self._get_pi_grow_legal_productions(
                node=node,
                max_depth=max_depth,
                overall_depth=overall_depth,
                recursive_in_queue=recursive_in_queue,
            )

            if not legal:
                raise RuntimeError(
                    f"No valid PI-Grow productions for symbol={node.symbol!r} "
                    f"at depth={node.depth} with max_depth={max_depth}."
                )

            must_force_growth = (overall_depth < max_depth) or (
                self._is_recursive_symbol(node.symbol) and not recursive_in_queue
            )

            if must_force_growth:
                production = rng.choice(legal)
            else:
                recursive = [p for p in legal if self._has_any_nonterminal(p)]
                terminating = [p for p in legal if self._is_terminal_only(p)]

                if recursive and terminating:
                    production = rng.choice(
                        recursive if rng.random() < 0.5 else terminating
                    )
                else:
                    production = rng.choice(legal)

            for symbol in production:
                child = TreeNode(symbol)
                node.add_child(child)

                if symbol in self.grammar.non_terminals:
                    queue.append(child)

        return root

    def generate_subtree(self, *, symbol: str, max_depth: int, rng: Any) -> "TreeNode":
        """
        Generate a subtree rooted at the given non-terminal symbol.
        Used by mutation.
        """
        effective_depth = min(max_depth, self.max_tree_depth)
        return self._generate_depthfirst_tree(
            start_symbol=symbol, max_depth=effective_depth, method="grow", rng=rng
        )

    def _generate_depthfirst_tree(
        self,
        *,
        max_depth: int,
        start_symbol: str,
        rng: Any,
        method: str,
        max_tries: int = 50,
        strict: bool = True,
    ) -> TreeNode:
        start_rule = self.grammar.rules[start_symbol]

        if method == "full" and start_rule.min_path is not None:
            if start_rule.min_path > max_depth:
                raise RuntimeError(
                    f"Full initialisation impossible: minimum derivation depth "
                    f"({start_rule.min_path}) exceeds max_depth ({max_depth})."
                )

        last_error: Exception | None = None

        for _ in range(max_tries):
            root = TreeNode(start_symbol)
            try:
                self._expand_depthfirst(
                    node=root,
                    max_depth=max_depth,
                    method=method,
                    strict=strict,
                    rng=rng,
                )
                return root
            except RuntimeError as e:
                last_error = e
                continue

        msg = (
            f"Failed to generate derivation tree using {method} initialisation.\n"
            f"max_depth={max_depth}, "
            f"min_path={start_rule.min_path}, "
            f"max_path={'∞' if start_rule.max_path is None else start_rule.max_path}."
        )
        raise RuntimeError(msg) from last_error

    def _expand_depthfirst(
        self,
        *,
        node: TreeNode,
        max_depth: int,
        method: str,
        strict: bool,
        rng: Any,
    ) -> None:
        # Terminals are already fully expanded.
        if node.symbol not in self.grammar.non_terminals:
            return

        legal = self._get_depthfirst_legal_productions(
            node=node,
            max_depth=max_depth,
            method=method,
            strict=strict,
        )

        if method == "grow":
            recursive = [p for p in legal if self._has_any_nonterminal(p)]
            terminating = [p for p in legal if self._is_terminal_only(p)]

            if recursive and terminating:
                # Sensible-initialisation grow behaviour:
                # choose recursive vs terminating group with equal probability.
                production = rng.choice(
                    recursive if rng.random() < 0.5 else terminating
                )
            else:
                production = rng.choice(legal)

        elif method == "full":
            production = rng.choice(legal)

        else:
            raise ValueError(f"Unknown depth-first generation method: {method}")

        for sym in production:
            child = TreeNode(sym)
            node.add_child(child)

            self._expand_depthfirst(
                node=child,
                max_depth=max_depth,
                method=method,
                strict=strict,
                rng=rng,
            )

    def generate_tree_ptc2(
        self,
        target_size: int,
        rng: Any,
        start_symbol: str | None = None,
        max_depth: int | None = None,
    ) -> TreeNode:
        """
        Generate a derivation tree using Probabilistic Tree Creation 2 (PTC2).

        `target_size` is counted in non-terminal expansions. Without
        `max_depth` this follows the refined PTC2 variant described by
        Nicolau (2017); with `max_depth` it uses the depth-limited PTC2D
        variant.

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

        # Start with the root as the only open non-terminal.
        root = TreeNode(start_symbol)

        # Open non-terminals that still need to be expanded.
        frontier: list[TreeNode] = [root]
        expansions_done = 0

        min_expansions = self._compute_min_expansions()

        # Minimum expansions needed to make a symbol terminal.
        def get_min_expansions(sym: str) -> int:
            if sym not in grammar.non_terminals:
                return 0

            value = min_expansions.get(sym)
            if value is None:
                raise RuntimeError(f"Symbol {sym!r} cannot be terminated.")

            return value

        # A smaller target could never produce a complete tree.
        target_size = max(target_size, get_min_expansions(start_symbol))

        # PTC2D uses minimum derivation depth to keep trees within max_depth.
        # Size feasibility uses minimum non-terminal expansion counts.
        def get_min_depth(sym: str) -> int:
            rule = grammar.rules.get(sym)
            return rule.min_path if (rule and rule.min_path is not None) else 0

        while frontier:
            # PTC2 samples from all open non-terminals instead of expanding
            # left-to-right; see Nicolau (2017), Section 3.4.
            node = frontier.pop(rng.randrange(len(frontier)))

            if node.symbol not in grammar.non_terminals:
                continue

            rule = grammar.rules[node.symbol]

            # Remaining expansion budget.
            RE = target_size - expansions_done

            # Minimum expansions needed to close the rest of the frontier.
            f_cost = sum(get_min_expansions(n.symbol) for n in frontier)

            # Keep productions that fit both the depth and size constraints.
            feasible = []
            for prod in rule.choices:
                # PTC2D only: enforce the derivation depth limit
                # separately from the PTC2 size budget.
                if max_depth is not None:
                    if node.depth + 1 > max_depth:
                        continue
                    # Can all children terminate before max_depth?
                    if any(
                        get_min_depth(s) + node.depth + 1 > max_depth
                        for s in prod
                        if s in grammar.non_terminals
                    ):
                        continue

                # Current expansion plus the minimum work left under its children.
                prod_cost = 1 + sum(
                    get_min_expansions(s) for s in prod if s in grammar.non_terminals
                )

                if prod_cost + f_cost <= RE:
                    feasible.append(prod)

            # Select the production.
            if not feasible:
                # If the size budget is exhausted, choose the least expansive production.
                production = min(
                    rule.choices,
                    key=lambda p: sum(
                        get_min_expansions(s) for s in p if s in grammar.non_terminals
                    ),
                )
            else:
                # Prefer recursion half the time while there is budget left.
                recursive = [
                    p for p in feasible if any(s in grammar.non_terminals for s in p)
                ]

                # Luke-style 50/50 group selection, also used in Nicolau's
                # refined PTC2 setup.
                if RE > 1 and recursive and rng.random() < 0.5:
                    production = rng.choice(recursive)
                else:
                    production = rng.choice(feasible)

            # Add the chosen production and keep any new non-terminals open.
            for sym in production:
                child = TreeNode(sym)
                node.add_child(child)
                if sym in grammar.non_terminals:
                    frontier.append(child)

            # Count this non-terminal expansion.
            expansions_done += 1

        return root

    def _is_terminal_only(self, production: list[str]) -> bool:
        return all(sym not in self.grammar.non_terminals for sym in production)

    def _has_any_nonterminal(self, production: list[str]) -> bool:
        return any(sym in self.grammar.non_terminals for sym in production)

    def _is_recursive_symbol(self, symbol: str) -> bool:
        """
        Check if the node is recursive
        PI-Grow cares about whether a queued node is recursive.
        """
        if symbol not in self.grammar.rules:
            return False
        return bool(self.grammar.rules[symbol].recursive)

    def _get_pi_grow_legal_productions(
        self,
        *,
        node: TreeNode,
        max_depth: int,
        overall_depth: int,
        recursive_in_queue: bool,
    ) -> list[list[str]]:
        """
        Return legal productions for PI-Grow.
        Before one branch reaches max_depth, PI-Grow behaves like Full:
        it prefers recursive growth. After that, it behaves like Grow.
        """
        rule = self.grammar.rules[node.symbol]
        productions_all = rule.choices

        # Children created by this production will be at node.depth + 1.
        # remaining_depth is max_depth - (node.depth + 1)
        remaining_depth = max_depth - (node.depth + 1)

        # If children would be placed at the maximum allowed depth, then only
        # terminal-only productions are allowed.
        if node.depth >= max_depth - 1:
            return [p for p in productions_all if self._is_terminal_only(p)]

        feasible = [
            p for p in productions_all if self._is_depth_feasible(p, remaining_depth)
        ]

        must_force_growth = (overall_depth < max_depth) or (
            self._is_recursive_symbol(node.symbol) and not recursive_in_queue
        )

        # If this node cannot grow, allow it to terminate; another queued
        # branch may still be able to reach the requested depth.
        if must_force_growth:
            growing = [p for p in feasible if self._has_any_nonterminal(p)]
            if growing:
                return growing
            return feasible

        recursive = [p for p in feasible if self._has_any_nonterminal(p)]
        terminating = [p for p in feasible if self._is_terminal_only(p)]

        if recursive and terminating:
            return recursive + terminating
        if recursive:
            return recursive
        if terminating:
            return terminating
        return []

    def _is_depth_feasible(self, production: list[str], remaining_depth: int) -> bool:
        for sym in production:
            if sym in self.grammar.non_terminals:
                child_rule = self.grammar.rules[sym]
                if child_rule.min_path is None:
                    return False
                if child_rule.min_path > remaining_depth:
                    return False
        return True

    def _compute_min_expansions(self) -> dict[str, int | None]:
        """
        Minimum non-terminal expansions needed to fully terminate each symbol.
        """

        min_exp: dict[str, int | None] = {nt: None for nt in self.grammar.non_terminals}

        changed = True
        while changed:
            changed = False

            for nt, rule in self.grammar.rules.items():
                candidates: list[int] = []

                for prod in rule.choices:
                    total = 1
                    valid = True

                    for sym in prod:
                        if sym in self.grammar.non_terminals:
                            child_min = min_exp[sym]
                            if child_min is None:
                                valid = False
                                break
                            total += child_min

                    if valid:
                        candidates.append(total)

                if candidates:
                    new_value = min(candidates)
                    if min_exp[nt] != new_value:
                        min_exp[nt] = new_value
                        changed = True

        return min_exp

    def _get_depthfirst_legal_productions(
        self,
        *,
        node: TreeNode,
        max_depth: int,
        method: str,
        strict: bool,
    ) -> list[list[str]]:
        """
        Return legal productions for depth-first Grow or Full generation.
        """

        rule = self.grammar.rules[node.symbol]
        productions_all = rule.choices

        remaining_depth = max_depth - (node.depth + 1)

        feasible = [
            p for p in productions_all if self._is_depth_feasible(p, remaining_depth)
        ]

        # Children created by this production will be at node.depth + 1.
        # remaining_depth is max_depth - (node.depth + 1)
        if node.depth >= max_depth - 1:
            candidates = [p for p in productions_all if self._is_terminal_only(p)]
            if not candidates:
                raise RuntimeError(
                    f"Depth limit reached for symbol={node.symbol!r} at depth={node.depth}, "
                    f"but no terminating productions exist under max_depth={max_depth}."
                )
            return candidates

        if method == "grow":
            recursive = [p for p in feasible if self._has_any_nonterminal(p)]
            terminating = [p for p in feasible if self._is_terminal_only(p)]

            if recursive and terminating:
                # Ryan/Azad sensible initialisation group-selection behaviour.
                return recursive + terminating
            if recursive:
                return recursive
            if terminating:
                return terminating
            raise RuntimeError(
                f"No valid Grow productions for symbol={node.symbol!r} "
                f"at depth={node.depth} under max_depth={max_depth}."
            )

        if method == "full":
            recursive_growth = [
                p
                for p in feasible
                if any(
                    sym in self.grammar.non_terminals
                    and self.grammar.rules[sym].recursive
                    for sym in p
                )
            ]
            nonterminal_growth = [p for p in feasible if self._has_any_nonterminal(p)]

            if recursive_growth:
                return recursive_growth
            if nonterminal_growth:
                return nonterminal_growth
            if not strict and feasible:
                return feasible

            raise RuntimeError(
                f"Full initialisation impossible for symbol={node.symbol!r} "
                f"at depth={node.depth} with max_depth={max_depth}."
            )

        raise ValueError(f"Unknown depth-first generation method: {method}")
