from typing import Any, Callable, List, Optional

from finchge.benchmarks.control.cartpole.environment import CartPoleEnvironment
from finchge.benchmarks.control.cartpole.interpreter import CartPoleInterpreter
from finchge.runners.control import ControlRunner


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
        # reject if passed empty phenotype
        if not phenotype or not phenotype.strip():
            return 0.0
        # Tokenize program once per episode
        tokens = self.interpreter.tokenize(phenotype)
        env.reset()
        total_reward = 0.0

        while not env.is_done():
            observation = env.get_observation()
            # start from the beginning pc=0
            action, _ = self.interpreter.evaluate(tokens, observation, 0)

            if action is None:
                break

            _, reward, _, _ = env.step(action)
            total_reward += reward
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

    @property
    def provided_context_keys(self) -> set[str]:
        return super().provided_context_keys | {"y_pred"}
