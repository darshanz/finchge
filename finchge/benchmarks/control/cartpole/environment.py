import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from finchge.benchmarks.control.base import ControlEnvironment


class CartPoleEnvironment(ControlEnvironment):
    def __init__(
        self,
        max_steps: int,
        gravity: float,
        mass_cart: float,
        mass_pole: float,
        length: float,
        force_mag: float,
        tau: float,
        theta_threshold_degrees: float,
        x_threshold: float,
        random_state: Optional[Any] = None,
    ):
        super().__init__(
            random_state=random_state, name="CartPole", max_steps=max_steps
        )

        # Physics Constants
        self.gravity = gravity
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.total_mass = mass_cart + mass_pole
        self.length = length  # actual length is twice
        self.pole_mass_length = mass_pole * length
        self.force_mag = force_mag
        self.tau = tau
        self.theta_threshold_radians = theta_threshold_degrees * 2 * np.pi / 360
        self.x_threshold = x_threshold

        # Normalization limits
        self.max_velocity = 5.0
        self.max_angular_velocity = 5.0

        self.reset()

    def reset(self) -> Dict[str, float]:
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
            return self.get_observation(), 0.0, True, {}

        force = self.force_mag if action == "right" else -self.force_mag
        costheta, sintheta = np.cos(self.theta), np.sin(self.theta)

        # Equations of motion
        temp = (
            force + self.pole_mass_length * self.theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        self.x += self.tau * self.x_dot
        self.x_dot += self.tau * xacc
        self.theta += self.tau * self.theta_dot
        self.theta_dot += self.tau * thetaacc

        self.steps_taken += 1
        self.episode_reward += 1.0
        self.done = (
            abs(self.x) > self.x_threshold
            or abs(self.theta) > self.theta_threshold_radians
            or self.steps_taken >= self.max_steps
        )

        return (
            self.get_observation(),
            1.0,
            self.done,
            {"x": self.x, "theta": self.theta},
        )

    def get_observation(self) -> Dict[str, float]:
        return {
            "cart_pos": np.clip(self.x / self.x_threshold, -1, 1),
            "cart_vel": np.clip(self.x_dot / self.max_velocity, -1, 1),
            "pole_angle": np.clip(self.theta / self.theta_threshold_radians, -1, 1),
            "pole_ang_vel": np.clip(self.theta_dot / self.max_angular_velocity, -1, 1),
        }

    def get_action_space(self) -> list[str]:
        return ["left", "right"]

    def get_available_actions(self) -> list[str]:
        return ["left", "right"]

    @classmethod
    def from_default(cls) -> "CartPoleEnvironment":
        base_path = Path(__file__).parent
        with open(base_path / "specs.json") as f:
            specs = json.load(f)
        specs.pop("name", None)  # not in __init__
        return cls(**specs)
