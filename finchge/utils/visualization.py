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

from finchge.grammar.derivation_tree import TreeNode


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


def plot_tree(tree_json: str, save_dir: str) -> None:
    """Plot individual trees from json files"""
    tree = TreeNode.from_string(tree_json)
    fig, ax = plt.subplots(figsize=(14, 10))

    def get_subtree_width(node: TreeNode) -> int:
        """Calculate width needed for subtree."""
        if not node.children:
            return 1
        return sum(get_subtree_width(child) for child in node.children)

    def process_node(
        node: TreeNode, x: Any, y: Any, width: float = 20, level: int = 0
    ) -> None:
        # color based on depth lighter color towards leaf nodes
        colormap = plt.colormaps["cool"]
        node_color = colormap(level / max(10, level + 1))

        # text color also based on nodes
        brightness = (
            0.299 * node_color[0] + 0.587 * node_color[1] + 0.114 * node_color[2]
        )
        text_color = "white" if brightness < 0.5 else "black"
        # for nodes just drawing a box around the text: adaptive font size according to depth level
        ax.text(
            x,
            y,
            str(node.symbol),
            ha="center",
            va="center",
            fontsize=max(8, 12 - level * 0.5),
            fontweight="bold",
            color=text_color,
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=node_color,
                edgecolor="black",
                linewidth=1,
                alpha=0.9,
            ),
        )

        if node.children:
            total_width = sum(get_subtree_width(child) for child in node.children)
            current_x = x - width / 2

            for child in node.children:
                child_width = get_subtree_width(child) * width / total_width
                child_x = current_x + child_width / 2
                child_y = (
                    y - 2.5
                )  # TODO Vertical spacing for now.. Not sure deeper trees will be ok

                # edge with gradient color, Edge thickness also changes with depth
                edge_color = plt.colormaps["Greys"](0.3 + level * 0.05)
                ax.plot(
                    [x, child_x],
                    [y, child_y],
                    color=edge_color,
                    linewidth=max(1, 2 - level * 0.2),
                    alpha=0.8,
                    zorder=2,
                )

                process_node(child, child_x, child_y, child_width, level + 1)
                current_x += child_width

    process_node(tree, 0, 0)

    # Adjust layout
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()

    output_path = f"{save_dir}/fittest_individual_tree.png"
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)

    print(f"Tree plot saved to: {output_path}")


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
