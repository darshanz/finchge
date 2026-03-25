from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.benchmarks.control.base import ControlEnvironment
from finchge.benchmarks.registry import register
from finchge.grammar import Grammar
from finchge.runners.control import ControlRunner


class CartPoleEnvironment(ControlEnvironment):
    """
    CartPole balancing environment

    Goal is to keep the pole upright.
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Observation: Normalized values for conditionals
    Reward: +1 per step alive
    """

    def __init__(
        self,
        max_steps: int = 500,
        gravity: float = 9.8,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        total_mass: Optional[float] = None,
        length: float = 0.5,
        force_mag: float = 10.0,
        tau: float = 0.02,
        theta_threshold_radians: float = 12 * 2 * np.pi / 360,
        random_state: Optional[Any] = None,
        x_threshold: float = 2.4,
    ):
        super().__init__(
            random_state=random_state, name="CartPole", max_steps=max_steps
        )

        self.gravity = gravity
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.total_mass = total_mass or (mass_cart + mass_pole)
        self.length = length
        self.pole_mass_length = mass_pole * length
        self.force_mag = force_mag
        self.tau = tau
        self.theta_threshold_radians = theta_threshold_radians
        self.x_threshold = x_threshold

        # For observation normalization
        self.max_angle = theta_threshold_radians
        self.max_position = x_threshold
        self.max_velocity = 5.0  # Approximate max velocity
        self.max_angular_velocity = 5.0  # Approximate max angular velocity

        self.state = None
        self.reset()

    def reset(self) -> Dict[str, float]:
        # Initialize with small random offsets
        self.x = self.rng.uniform(-0.05, 0.05)
        self.x_dot = self.rng.uniform(-0.05, 0.05)
        self.theta = self.rng.uniform(-0.05, 0.05)
        self.theta_dot = self.rng.uniform(-0.05, 0.05)

        self.steps_taken = 0
        self.episode_reward = 0.0
        self.done = False

        return self.get_observation()

    def step(self, action: str) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        if self.done:
            return self.get_observation(), 0.0, True, {"error": "Episode already done"}

        # Map action to force
        if action == "left":
            force = -self.force_mag
        elif action == "right":
            force = self.force_mag
        else:
            raise ValueError(f"Invalid action: {action}")

        # Physics equations (from classic CartPole)
        costheta = np.cos(self.theta)
        sintheta = np.sin(self.theta)

        temp = (
            force + self.pole_mass_length * self.theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass

        # Update state (Euler integration)
        self.x += self.tau * self.x_dot
        self.x_dot += self.tau * xacc
        self.theta += self.tau * self.theta_dot
        self.theta_dot += self.tau * thetaacc

        self.steps_taken += 1
        self.episode_reward += 1  # +1 per step

        # Check termination
        info: dict[str, Any] = {}

        if (
            abs(self.x) > self.x_threshold
            or abs(self.theta) > self.theta_threshold_radians
        ):
            self.done = True
            info["termination"] = "out_of_bounds"
        elif self.steps_taken >= self.max_steps:
            self.done = True
            info["termination"] = "max_steps"

        info["cart_position"] = self.x
        info["pole_angle"] = self.theta
        info["steps"] = self.steps_taken

        return self.get_observation(), 1.0, self.done, info

    def get_observation(self) -> Dict[str, float]:
        return {
            "cart_pos": np.clip(self.x / self.max_position, -1, 1),
            "cart_vel": np.clip(self.x_dot / self.max_velocity, -1, 1),
            "pole_angle": np.clip(self.theta / self.max_angle, -1, 1),
            "pole_ang_vel": np.clip(self.theta_dot / self.max_angular_velocity, -1, 1),
        }

    def get_raw_state(self) -> NDArray[np.float64]:
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])

    def get_available_actions(self) -> list[str]:
        return ["left", "right"]

    def get_action_space(self) -> list[str]:
        return ["left", "right"]


# Interpreter
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
        if pc >= len(tokens):
            return None, pc

        token = tokens[pc]

        # Actions
        if token in self.actions:
            return token, pc + 1

        if token == "if":
            if pc + 1 >= len(tokens) or tokens[pc + 1] != "(":
                return None, pc + 2
            condition_tokens = []
            cond_pc = pc + 2
            while cond_pc < len(tokens) and tokens[cond_pc] != ")":
                condition_tokens.append(tokens[cond_pc])
                cond_pc += 1
            if cond_pc >= len(tokens) or tokens[cond_pc] != ")":
                return None, cond_pc
            condition_true = self.evaluate_condition(condition_tokens, observation)

            if cond_pc + 1 >= len(tokens) or tokens[cond_pc + 1] != "(":
                return None, cond_pc + 2
            true_action, true_pc = self.evaluate(tokens, observation, cond_pc + 2)

            if true_pc >= len(tokens) or tokens[true_pc] != ")":
                return None, true_pc

            if true_pc + 1 >= len(tokens) or tokens[true_pc + 1] != "else":
                return None, true_pc + 2

            if true_pc + 2 >= len(tokens) or tokens[true_pc + 2] != "(":
                return None, true_pc + 3

            false_action, false_pc = self.evaluate(tokens, observation, true_pc + 3)

            if false_pc >= len(tokens) or tokens[false_pc] != ")":
                return None, false_pc

            if condition_true:
                return true_action, false_pc + 1
            else:
                return false_action, false_pc + 1

        return self.evaluate(tokens, observation, pc + 1)


# Cart-Pole Runner
class CartPoleRunner(ControlRunner[CartPoleEnvironment]):
    def __init__(
        self,
        env_factory: Callable[[], CartPoleEnvironment],
        n_episodes: int = 1,
        interpreter: Optional[CartPoleInterpreter] = None,
        optimal_fitness: Optional[float] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        super().__init__(
            random_state=random_state,
            env_factory=env_factory,
            n_episodes=n_episodes,
            optimal_fitness=optimal_fitness,
        )
        self.interpreter = interpreter or CartPoleInterpreter()
        self._last_rewards: List[float] = []

    def _run_episode(self, phenotype: str, env: CartPoleEnvironment) -> float:
        # Tokenize program once per episode
        tokens = self.interpreter.tokenize(phenotype)

        env.reset()
        program_counter = 0
        total_reward = 0.0

        while not env.is_done():
            observation = env.get_observation()
            action, program_counter = self.interpreter.evaluate(
                tokens, observation, program_counter
            )

            if action is None:
                actions = env.get_available_actions()
                action = self.rng.choice(actions) if actions else None
                program_counter = 0

            if action:
                _, reward, _, _ = env.step(action)
                total_reward += reward

            if program_counter >= len(tokens):
                program_counter = 0

        return total_reward

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.update(
            {
                "interpreter": self.interpreter,
                "_last_rewards": self._last_rewards,
            }
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


# Cart-Pole Benchmark
def create_cartpole_environment(max_steps: int = 500) -> CartPoleEnvironment:
    return CartPoleEnvironment(max_steps=max_steps)


@register("cartpole", "control")
class CartPoleBenchmark(Benchmark):
    def __init__(
        self,
        random_state: Optional[Any] = None,
        max_steps: int = 500,
        n_episodes: int = 1,
    ) -> None:
        super().__init__(random_state=random_state)
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.name = "Cart-Pole"

        self._metadata = BenchmarkMetadata(
            name="Cart-Pole",
            category="control",
            description="...",
            reference="...",
            input_dim=0,
            output_dim=1,
            train_size=n_episodes,
            test_size=n_episodes,
        )

    def _generate_data(
        self,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        raise NotImplementedError(
            "Control Problems do not implement _generate_data function."
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def grammar_str(self) -> str:
        return """
        <code> ::= <line> | <code> <line>
        <line> ::= <if> | <action>
        <if> ::= if ( <condition> ) <line> else <line>
        <condition> ::= <state> <op> <number>
        <state> ::= cart_pos | cart_vel | pole_angle | pole_ang_vel
        <op> ::= < | > | <= | >=
        <number> ::= -1.0 | -0.5 | 0.0 | 0.5 | 1.0
        <action> ::= left | right
        """

    def grammar(self) -> Grammar:
        """
        Return the Grammar object.
        """
        return Grammar(grammar_str=self.grammar_str())

    def create_runner(self, data_type: str = "train") -> CartPoleRunner:
        def env_factory() -> CartPoleEnvironment:
            return create_cartpole_environment(self.max_steps)

        return CartPoleRunner(env_factory, n_episodes=self.n_episodes)
