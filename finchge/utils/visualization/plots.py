import logging
import os
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray


def plot_fitness_chart(
    data: list[float],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    objective_name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    if objective_name:
        ylabel = f"Fitness value ({objective_name})"

    title = "Fitness" if not title else title
    xlabel = "Generation" if not xlabel else xlabel
    ylabel = "Fitness value" if not ylabel else ylabel
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_best_fitness(
    best_fitness: Union[list[Any], NDArray[np.float64]],
    title: str = "Fitness Progress",
    x_label: str = "Generation",
    y_label: str = "Fitness",
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """
    Plot the best fitness across generations.

    Args:
        best_fitness: List or array of best fitness values per generation
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Tuple of (fig, ax)
    """
    best_fitness = np.array(best_fitness)
    generations = np.arange(1, len(best_fitness) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(generations, best_fitness, "b-", linewidth=1.5, label="Best Fitness")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def visualize_pareto_front(save_dir: str, objective_names: list[str]) -> None:
    if len(objective_names) < 2:
        raise ValueError("objective_names must contain at least two objectives")

    pareto_csv_path: str = f"{save_dir}/pareto_front.csv"
    df: pd.DataFrame = pd.read_csv(pareto_csv_path)

    os.makedirs(save_dir, exist_ok=True)

    # Extract generations safely
    generations = df["generation"]
    unique_gens: list[int] = sorted(int(g) for g in generations.unique())

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Colormap colors
    cmap = plt.colormaps["viridis"]
    colors = cmap(np.linspace(0.0, 1.0, len(unique_gens)))

    # Scatter all solutions
    all_data: pd.DataFrame = df[objective_names + ["generation"]]

    ax1.scatter(
        all_data[objective_names[0]],
        all_data[objective_names[1]],
        c=all_data["generation"],
        cmap="viridis",
        alpha=0.2,
        s=15,
        marker=".",
    )

    # Plot Pareto fronts per generation
    for i, gen in enumerate(unique_gens):
        gen_data: pd.DataFrame = df[df["generation"] == gen]
        gen_data_sorted: pd.DataFrame = gen_data.sort_values(by=objective_names[0])

        ax1.step(
            gen_data_sorted[objective_names[0]],
            gen_data_sorted[objective_names[1]],
            linestyle="--",
            where="pre",
            color=colors[i],
            lw=0.35,
            alpha=0.25,
        )

        ax1.plot(
            gen_data_sorted[objective_names[0]],
            gen_data_sorted[objective_names[1]],
            "o",
            color=colors[i],
            ms=1,
        )

    ax1.set_xlabel(objective_names[0], fontsize=14)
    ax1.set_ylabel(objective_names[1], fontsize=14)
    ax1.set_title("Pareto Fronts by Generation")

    # Colorbar
    norm = Normalize(vmin=0, vmax=len(unique_gens) - 1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax1, ticks=[0, len(unique_gens) - 1])
    cbar.ax.set_ylabel("Generation", rotation=90)

    plt.tight_layout()

    plt.savefig(f"{save_dir}/pareto_front_evolution.pdf")
    plt.savefig(f"{save_dir}/pareto_front_evolution.png", dpi=300)
    plt.close(fig)

    logging.info(
        f"Pareto front evolution plot saved to: {save_dir}/pareto_front_evolution.pdf"
    )
