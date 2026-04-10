import re
from abc import ABC, abstractmethod
from typing import Optional

from finchge.grammar import Rule
from finchge.grammar.range_handlers import RangeHandlerRegistry


class GrammarParser(ABC):
    """
    Base class for all grammar parsers.

    Any custom parser must implement the parse() method which returns
    a standardized tuple containing grammar components.
    """

    @abstractmethod
    def parse(
        self,
    ) -> tuple[dict[str, "Rule"], dict[str, "Rule"], str, list[str], list[str]]:
        """
        Parse input data and return grammar components.

        Returns:
            tuple : containing the rules, terminals and non-terminals

        Response Contains:
            - `rules_original` (dict[str, Rule]): Mapping of non-terminal symbols to Rule objects (original written form without range expansion).
            - `rules` (dict[str, Rule]): Mapping of non-terminal symbols to Rule objects in Expanded (range notation) form.
            - `start_rule` (str): The first non-terminal rule, considered the start rule.
            - `terminals` (set[str]): Set of terminal symbols.
            - `non_terminals` (set[str]): Set of non-terminal symbols.

        """
        pass


def remove_comments(lines: list[str]) -> list[str]:
    """
    Remove comments and empty lines.
    """
    COMMENT_CHAR = "#"
    cleaned = []
    for line in lines:
        # Strip whitespace
        stripped = line.strip()
        if not stripped:  # empty lines
            cleaned.append("")

        # Remove comments
        if COMMENT_CHAR in stripped:
            stripped = stripped.split(COMMENT_CHAR, 1)[0].rstrip()
            if not stripped:  # Whole line was a comment
                continue

        cleaned.append(stripped)
    return cleaned


def consolidate_multiline_rules(lines: list[str]) -> list[str]:
    """
    If same rule sreads across multiple lines, then just make them a single line
    Simple rule is unti we see a token before ::= , its the same rule

    Args:
        lines:

    Returns:

    """
    consolidated: list[str] = []
    current_rule: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue  # Skip truly empty lines

        # new rule is detected based on presence of  "::=".
        # Until it appears it's the same rule, no matter how many line breaks it has
        if "::=" in stripped:
            #  building previous rule?
            if current_rule:
                consolidated.append(" ".join(current_rule))
                current_rule = []
            # new rule with this line
            current_rule.append(stripped)
        else:
            # continuation line.just append to current rule.
            if not current_rule:
                raise ValueError(f"Line '{line}' has no rule to continue.")
            current_rule.append(stripped)
    # last rule
    if current_rule:
        consolidated.append(" ".join(current_rule))

    return consolidated


def split_choices(rhs_symbols: list[str]) -> list[list[str]]:
    """
    Split RHS  into production choices
    For now it simply finds choices based on presence of choice symbol (Can be improved later, if needed.)
    """
    # placeholder variables
    production_choices: list[list[str]] = []
    current_symbol: list[str] = []
    # just iterate in rhs symbols if choice symbol (pipe) is found
    for item in rhs_symbols:
        if item == "|":
            if current_symbol:
                production_choices.append(current_symbol)
                current_symbol = []
        else:
            current_symbol.append(item)
    if current_symbol:
        production_choices.append(current_symbol)
    return production_choices


class BNFGrammarParser(GrammarParser):
    """
    Parses a grammar string written in Backus-Naur Form (BNF) into structured rules.

    The parser identifies terminal and non-terminal symbols, splits productions into alternatives,
    and validates the structure of the grammar.

    Attributes:
        grammar_str (str): The raw BNF grammar string.
    Args:
        grammar_str (str): The grammar definition in BNF format.

    """

    def __init__(self, grammar_str: str) -> None:
        super().__init__()
        self.rules_original: dict[str, "Rule"] = {}
        self.rules: dict[str, "Rule"] = {}
        self.non_terminals: dict[str, None] = {}
        self.terminals: dict[str, None] = {}
        self.start_rule: Optional[str] = None
        self.grammar_str: str = grammar_str

        # Setup
        self.non_terminal_pattern = re.compile(r"^<\w+>$")

        # Registry for registering range handlers
        self.range_registry = RangeHandlerRegistry()

    def _scan_rhs(self, rhs: str) -> list[str]:
        # spaces around the pipe are also not important
        rhs = re.sub(r"\s*\|\s*", "|", rhs)

        tokens = []
        i = 0
        n = len(rhs)

        # first space in the RHS is not important remove that
        if i < n and rhs[i].isspace():
            i += 1

        while i < n:
            ch = rhs[i]
            if ch == "|":
                tokens.append("|")
                i += 1
                continue
            if ch == '"':
                j = rhs.find('"', i + 1)
                if j == -1:
                    raise ValueError(f"Unclosed double quote at {i}")
                tokens.append(rhs[i + 1 : j])  # strip quotes
                i = j + 1
                continue
            if ch == "'":
                j = rhs.find("'", i + 1)
                if j == -1:
                    raise ValueError(f"Unclosed single quote at {i}")
                tokens.append(rhs[i + 1 : j])  # strip quotes
                i = j + 1
                continue
            if ch == "<":
                j = i + 1
                while j < n and rhs[j] not in ("|", "\n"):
                    if rhs[j] == ">":
                        tokens.append(rhs[i : j + 1])
                        i = j + 1
                        break
                    j += 1
                else:
                    # no matching '>', treat as plain terminal
                    pass
                if i > j:
                    continue
            # Plain terminal: collect until '|' or start of quote or non-terminal
            start = i
            while i < n and rhs[i] != "|":
                if rhs[i] in ('"', "'"):
                    break
                if rhs[i] == "<":
                    # check for valid non-terminal
                    k = i + 1
                    while k < n and rhs[k] not in ("|", "\n"):
                        if rhs[k] == ">":
                            break
                        k += 1
                    else:
                        i += 1
                        continue
                    break
                i += 1
            token = rhs[start:i]
            if token:
                tokens.append(token)
        return tokens

    def _preprocess_ranges(self, rhs: str) -> tuple[str, dict[str, list[str]]]:
        """
        Until other symbols are handled we just use placeholders for the range handlers.
        The ranges could be directly replaced in the beginning, if the ranges are placed only at the end of the grammar
        To avoid worst case scenario in grammar design, replacing range constructs with placeholders.
        Returns (modified_rhs, placeholder_map).
        """
        range_regex = self.range_registry.get_combined_pattern()
        if not range_regex:
            return rhs, {}

        placeholders = {}
        counter = 0

        def replacer(match: re.Match[str]) -> str:
            nonlocal counter
            token: str = match.group(0)
            expanded = self.range_registry.expand_token(token)
            if expanded and expanded != [token]:
                placeholder = f"__RANGE_{counter}__"
                counter += 1
                placeholders[placeholder] = expanded
                return placeholder
            return token

        modified = re.sub(range_regex, replacer, rhs)
        return modified, placeholders

    def parse(
        self,
    ) -> tuple[dict[str, Rule], dict[str, Rule], str, list[str], list[str]]:
        """
        Parses the BNF grammar string into rules, terminals, and non-terminals.

        Returns:
            Tuple: Tuple containing rules (original), rules_expanded, start_rule, terminals and non-terminals.

        Response contains:
            - `rules_original` (dict[str, Rule]): Mapping of non-terminal symbols to Rule objects. (Original may be contracted form)
            - `rules` (dict[str, Rule]): Mapping of non-terminal symbols to Rule objects in Expanded (range notation) form.
            - `start_rule` (str): The first non-terminal rule, considered the start rule.
            - `terminals` (set[str]): List of terminal symbols.
            - `non_terminals` (set[str]): List of non-terminal symbols.


        Raises:
            ValueError: If the grammar syntax is invalid or contains undefined symbols.
        """
        # get all lines from the grammar
        lines = self.grammar_str.strip().split("\n")
        #  filter the comments
        filtered_lines = remove_comments(lines)
        # Consolidate multi-line rules
        consolidated_lines = consolidate_multiline_rules(filtered_lines)

        rhs_tokens = []

        for line in consolidated_lines:
            if "::=" in line:
                lhs, rhs = line.split("::=", 1)
                lhs = lhs.strip()

                # Validate LHS
                if not self.non_terminal_pattern.match(lhs):
                    raise ValueError(f"Value Error: Invalid non-terminal '{lhs}'")

                # first expand the ranges
                rhs_processed, range_placeholders = self._preprocess_ranges(rhs)
                tokens = self._scan_rhs(rhs_processed)

                # Process tokens using the handler registry
                expanded_tokens = []
                for token in tokens:
                    token_clean = token.strip()
                    if token_clean in range_placeholders:
                        expanded_tokens.extend(range_placeholders[token_clean])
                    else:
                        expanded_tokens.append(token)

                # Store for later validation
                rhs_tokens.append((lhs, expanded_tokens))

                # Create rules
                original_choices = split_choices(tokens)
                expanded_choices = split_choices(expanded_tokens)

                self.rules_original[lhs] = Rule(lhs, original_choices)
                self.rules[lhs] = Rule(lhs, expanded_choices)
                self.non_terminals.setdefault(lhs, None)

        # Validate all non-terminals are defined
        all_defined_non_terminals = self.non_terminals

        for lhs, tokens_ in rhs_tokens:
            for token in tokens_:
                if token == "|":
                    continue

                if self.non_terminal_pattern.match(token):
                    if token not in all_defined_non_terminals:
                        raise ValueError(
                            f"Undefined non-terminal symbol '{token}', found in rule for '{lhs}'"
                        )
                else:
                    self.terminals.setdefault(token, None)

        self.start_rule = None
        for line in consolidated_lines:
            if "::=" in line:
                lhs = line.split("::=", 1)[0].strip()
                if lhs in self.non_terminals:
                    self.start_rule = lhs
                    break

        # This should never happen, but let's be defensive
        if self.start_rule is None:
            # Fallback: use first non-terminal
            self.start_rule = next(iter(self.non_terminals.keys()))

        return (
            self.rules_original,
            self.rules,
            self.start_rule,
            list(self.terminals.keys()),
            list(self.non_terminals.keys()),
        )
