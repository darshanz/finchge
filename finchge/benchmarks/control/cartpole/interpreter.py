from typing import Dict, List, Optional, Tuple


class CartPoleInterpreter:
    """
    Interpreter for cartpole balancing programs.
    """

    def __init__(self) -> None:
        self.actions = ["left", "right"]
        self.state_vars = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
        self.operators = ["<", ">", "<=", ">="]
        self.numbers = ["-1.0", "-0.5", "0.0", "0.5", "1.0"]

    def tokenize(self, program: str) -> List[str]:
        tokens = []
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

            keywords = self.actions + self.state_vars + self.operators + ["if", "else"]
            for kw in sorted(keywords, key=len, reverse=True):
                if program.startswith(kw, i):
                    tokens.append(kw)
                    i += len(kw)
                    break
            else:
                # Check for numbers
                num_match = False
                for num in self.numbers:
                    if program.startswith(num, i):
                        tokens.append(num)
                        i += len(num)
                        num_match = True
                        break

                if not num_match:
                    # Unknown token, skip one character
                    i += 1

        return tokens

    def evaluate_condition(
        self, condition_tokens: List[str], observation: Dict[str, float]
    ) -> bool:
        if len(condition_tokens) != 3:
            return False

        var, op, num_str = condition_tokens
        if var not in self.state_vars:
            return False
        if op not in self.operators:
            return False
        try:
            num = float(num_str)
        except ValueError:
            return False
        value = observation.get(var, 0.0)

        if op == "<":
            return value < num
        elif op == "<=":
            return value <= num
        elif op == ">":
            return value > num
        elif op == ">=":
            return value >= num
        else:
            return False

    def evaluate(
        self, tokens: List[str], observation: Dict[str, float], pc: int = 0
    ) -> Tuple[Optional[str], int]:
        while pc < len(tokens):
            token = tokens[pc]

            # Actions
            if token in self.actions:
                return token, pc + 1

            if token == "if":
                try:
                    # the format should be
                    # if ( <condition> ) ( <code> ) else ( <code> )
                    if pc + 1 >= len(tokens) or tokens[pc + 1] != "(":
                        return None, pc + 1

                    # Extract condition tokens
                    cond_start = pc + 2
                    cond_end = cond_start + 3

                    if cond_end >= len(tokens) or tokens[cond_end] != ")":
                        return None, cond_end

                    condition_tokens = tokens[cond_start:cond_end]

                    condition_true = self.evaluate_condition(
                        condition_tokens, observation
                    )

                    true_branch_open = cond_end + 1
                    if (
                        true_branch_open >= len(tokens)
                        or tokens[true_branch_open] != "("
                    ):
                        return None, true_branch_open

                    true_action, true_end = self.evaluate(
                        tokens, observation, true_branch_open + 1
                    )

                    if true_end >= len(tokens) or tokens[true_end] != ")":
                        return None, true_end

                    else_index = true_end + 1
                    if else_index >= len(tokens) or tokens[else_index] != "else":
                        return None, else_index

                    false_branch_open = else_index + 1
                    if (
                        false_branch_open >= len(tokens)
                        or tokens[false_branch_open] != "("
                    ):
                        return None, false_branch_open

                    false_action, false_end = self.evaluate(
                        tokens, observation, false_branch_open + 1
                    )

                    if false_end >= len(tokens) or tokens[false_end] != ")":
                        return None, false_end

                    final_pc = false_end + 1

                    if condition_true:
                        return true_action, final_pc
                    else:
                        return false_action, final_pc

                except IndexError:
                    return None, pc + 1

            # Skip unknown tokens
            pc += 1

        return None, pc
