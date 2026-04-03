from typing import Any

import matplotlib.pyplot as plt

from finchge.grammar.derivation_tree import TreeNode


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
