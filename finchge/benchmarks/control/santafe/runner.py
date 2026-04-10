from typing import Any, Callable, Dict, Optional

from finchge.benchmarks.control.santafe.environment import SantaFeEnvironment
from finchge.benchmarks.control.santafe.interpreter import SantaFeInterpreter
from finchge.runners import ControlRunner


class SantaFeRunner(ControlRunner[SantaFeEnvironment]):
    """
    Runner for Santa Fe Trail problem.
    """

    def __init__(
        self,
        env_factory: Callable[[], SantaFeEnvironment],
        interpreter: Optional[SantaFeInterpreter] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        """
        Initialize Santa Fe runner.

        Args:
            env_factory: Function that creates new SantaFeEnvironment instances
            interpreter: Program interpreter (creates new one if None)
        """
        super().__init__(
            random_state=random_state,
            env_factory=env_factory,
            n_episodes=1,
            optimal_fitness=89.0,
        )
        self.interpreter = interpreter or SantaFeInterpreter()

    def _run_episode(self, phenotype: str, env: SantaFeEnvironment) -> float:
        """Run one episode in Santa Fe environment."""
        # Tokenize program
        tokens = self.interpreter.tokenize(phenotype)

        env.reset()
        program_counter = 0
        total_reward = 0.0

        while not env.is_done():
            food_ahead = env.get_observation()

            action, program_counter = self.interpreter.evaluate(
                tokens, food_ahead, program_counter
            )

            if action is None:
                break

            if action:
                _, reward, done, _ = env.step(action)
                total_reward += reward

            # Wrap program counter if at end
            if program_counter >= len(tokens):
                program_counter = 0

        return total_reward

    def __getstate__(self) -> Dict[str, Any]:
        """Called when pickling."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Called when unpickling."""
        self.__dict__.update(state)

    def evaluate(self, phenotype: str) -> float:
        """
        Evaluate SantaFe
        Overrides the base evaluate because santafe needs to run a single eposide only.
        Args:
            phenotype:

        Returns:

        """
        env = self.env_factory()
        return self._run_episode(phenotype, env)

    @property
    def provided_context_keys(self) -> set[str]:
        return super().provided_context_keys | {"y_pred"}
