from typing import Dict, List, Optional, Tuple


class MazeInterpreter:
    """
    Interpreter for maze navigation programs.

    """

    def __init__(self) -> None:
        self.actions: List[str] = ["up", "down", "left", "right"]
        self.conditions: List[str] = ["if-wall-ahead", "if-wall-left", "if-wall-right"]
        self.keywords: List[str] = self.actions + self.conditions

    def tokenize(self, program: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        while i < len(program):
            # Skip whitespace
            if program[i].isspace():
                i += 1
                continue

            # Parentheses as separate tokens
            if program[i] == "(":
                tokens.append("(")
                i += 1
                continue
            if program[i] == ")":
                tokens.append(")")
                i += 1
                continue

            # Check for conditions (longest first)
            matched = False
            for cond in sorted(self.conditions, key=len, reverse=True):
                if program.startswith(cond, i):
                    tokens.append(cond)
                    i += len(cond)
                    matched = True
                    break

            if matched:
                continue

            # Check for actions
            for action in self.actions:
                if program.startswith(action, i):
                    tokens.append(action)
                    i += len(action)
                    matched = True
                    break

            if not matched:
                # Unknown token, skip one character
                i += 1

        return tokens

    def evaluate(
        self, tokens: List[str], observation: Dict[str, bool], pc: int = 0
    ) -> Tuple[Optional[str], int]:
        if pc >= len(tokens):
            return None, pc

        token = tokens[pc]

        # Actions
        if token in self.actions:
            return token, pc + 1

        # Wall conditionals
        if token in self.conditions:
            # which direction to check
            wall_present: bool = False
            if token == "if-wall-ahead":
                wall_present = observation.get("up", False)
            elif token == "if-wall-left":
                wall_present = observation.get("left", False)
            elif token == "if-wall-right":
                wall_present = observation.get("right", False)
            if pc + 1 >= len(tokens) or tokens[pc + 1] != "(":
                return None, pc + 2

            true_action, true_pc = self.evaluate(tokens, observation, pc + 2)
            if true_pc >= len(tokens) or tokens[true_pc] != ")":
                return None, true_pc
            if true_pc + 1 >= len(tokens) or tokens[true_pc + 1] != "(":
                return None, true_pc + 2
            false_action, false_pc = self.evaluate(tokens, observation, true_pc + 2)
            if false_pc >= len(tokens) or tokens[false_pc] != ")":
                return None, false_pc
            if wall_present:
                return true_action, false_pc + 1
            else:
                return false_action, false_pc + 1
        return self.evaluate(tokens, observation, pc + 1)
