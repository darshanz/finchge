import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_single_objective_fitness(save_dir: str) -> None:
    csv_path = f"{save_dir}/generations.csv"
    df = pd.read_csv(csv_path)

    if "gen" not in df.columns or "best_fitness" not in df.columns:
        raise ValueError("generations.csv must contain 'gen' and 'best_fitness'")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["gen"], df["best_fitness"], label="Best Fitness", linewidth=2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fitness_progression.pdf")
    plt.close(fig)


def plot_single_objective_complexity(save_dir: str) -> None:
    csv_path = f"{save_dir}/generations.csv"
    df = pd.read_csv(csv_path)

    if "gen" not in df.columns:
        raise ValueError("generations.csv must contain 'gen'")

    complexity_cols = [
        ("ave_genome_length", "Average Genome Length"),
        ("ave_tree_depth", "Average Tree Depth"),
        ("ave_tree_nodes", "Average Tree Nodes"),
        ("ave_used_codons", "Average Used Codons"),
        ("best_genome_length", "Best Genome Length"),
        ("best_tree_depth", "Best Tree Depth"),
        ("best_tree_nodes", "Best Tree Nodes"),
        ("best_used_codons", "Best Used Codons"),
    ]

    available = [(col, label) for col, label in complexity_cols if col in df.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for col, label in available:
        ax.plot(df["gen"], df[col], label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Complexity")
    ax.set_title("Complexity Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/complexity_progression.pdf")
    plt.close(fig)


def plot_search_diagnostics(save_dir: str) -> None:
    csv_path = f"{save_dir}/generations.csv"
    df = pd.read_csv(csv_path)

    if "gen" not in df.columns:
        raise ValueError("generations.csv must contain 'gen'")

    cols = [
        ("unique_inds", "Unique Individuals"),
        ("unused_search", "Unused Search (%)"),
        ("invalids", "Invalid Individuals"),
    ]
    available = [(col, label) for col, label in cols if col in df.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for col, label in available:
        ax.plot(df["gen"], df[col], label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Count / Percentage")
    ax.set_title("Search Diagnostics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/search_diagnostics.pdf")
    plt.close(fig)


def plot_generation_runtime(save_dir: str) -> None:
    csv_path = f"{save_dir}/generations.csv"
    df = pd.read_csv(csv_path)

    if "gen" not in df.columns or "time_taken" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["gen"], df["time_taken"], linewidth=1.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Seconds")
    ax.set_title("Generation Runtime")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/generation_runtime.pdf")
    plt.close(fig)


def visualize_final_pareto_front(save_dir: str, objective_names: list[str]) -> None:
    if len(objective_names) < 2:
        raise ValueError("Need at least two objectives to plot a Pareto front")

    csv_path = f"{save_dir}/final_pareto_front/front.csv"
    df = pd.read_csv(csv_path)

    x_name = objective_names[0]
    y_name = objective_names[1]

    if x_name not in df.columns or y_name not in df.columns:
        raise ValueError(f"Expected columns '{x_name}' and '{y_name}' in {csv_path}")

    plot_df = df[[x_name, y_name]].dropna().sort_values(by=x_name)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(plot_df[x_name], plot_df[y_name], s=25, alpha=0.8)
    ax.plot(plot_df[x_name], plot_df[y_name], alpha=0.7)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title("Final Pareto Front")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/final_pareto_front/final_pareto_front.pdf")
    plt.close(fig)


def visualize_pareto_front_evolution(save_dir: str, objective_names: list[str]) -> None:
    if len(objective_names) < 2:
        raise ValueError("Need at least two objectives to plot Pareto front evolution")

    base_dir = Path(save_dir)
    front_files = sorted(base_dir.glob("generation_*/front.csv"))

    if not front_files:
        raise FileNotFoundError(f"No generation front.csv files found in {save_dir}")

    frames: list[pd.DataFrame] = []

    for csv_path in front_files:
        match = re.search(r"generation_(\d+)", str(csv_path.parent))
        if not match:
            continue

        generation = int(match.group(1))
        df = pd.read_csv(csv_path)

        x_name = objective_names[0]
        y_name = objective_names[1]

        if x_name not in df.columns or y_name not in df.columns:
            continue

        df = df.copy()
        df["generation"] = generation
        frames.append(df)

    if not frames:
        raise ValueError(
            "No valid front.csv files found with the required objective columns"
        )

    all_df = pd.concat(frames, ignore_index=True)

    x_name = objective_names[0]
    y_name = objective_names[1]
    unique_gens = sorted(all_df["generation"].unique())

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.colormaps["viridis"]
    norm = Normalize(vmin=min(unique_gens), vmax=max(unique_gens))

    ax.scatter(
        all_df[x_name],
        all_df[y_name],
        c=all_df["generation"],
        cmap=cmap,
        norm=norm,
        alpha=0.25,
        s=15,
        marker=".",
    )

    for gen in unique_gens:
        gen_df = all_df[all_df["generation"] == gen].sort_values(by=x_name)

        ax.step(
            gen_df[x_name],
            gen_df[y_name],
            where="pre",
            linestyle="--",
            lw=0.7,
            alpha=0.35,
            color=cmap(norm(gen)),
        )
        ax.plot(
            gen_df[x_name],
            gen_df[y_name],
            "o",
            ms=1.5,
            alpha=0.6,
            color=cmap(norm(gen)),
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title("Pareto Front Evolution")
    ax.grid(True, alpha=0.3)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel("Generation", rotation=90)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/pareto_front_evolution.pdf")
    plt.close(fig)


def plot_front_size(save_dir: str) -> None:
    csv_path = f"{save_dir}/generations.csv"
    df = pd.read_csv(csv_path)

    if "gen" not in df.columns or "front_size" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["gen"], df["front_size"], linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Front Size")
    ax.set_title("Pareto Front Size Over Generations")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/front_size_progression.pdf")
    plt.close(fig)


def plot_objective_progress(save_dir: str, objective_names: list[str]) -> None:
    csv_path = f"{save_dir}/generations.csv"
    df = pd.read_csv(csv_path)

    if "gen" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False

    for name in objective_names:
        mean_col = f"front_mean_{name}"
        min_col = f"front_min_{name}"

        if mean_col in df.columns:
            ax.plot(df["gen"], df[mean_col], label=f"{name} mean")
            plotted = True

        if min_col in df.columns:
            ax.plot(df["gen"], df[min_col], label=f"{name} min", linestyle="--")
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Generation")
    ax.set_ylabel("Objective Value")
    ax.set_title("Objective-wise Pareto Front Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/objective_progress.pdf")
    plt.close(fig)
