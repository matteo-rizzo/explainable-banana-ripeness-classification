from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.tree import DecisionTreeClassifier

from scripts.decision_tree import get_leaf_constraints, train_dt


def explode(data):
    size = np.array(data.shape) * 2
    if len(size) > 3:
        size = np.array([*size[:-1], data.shape[-1] + 1])
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def create_bins_range(value_range: Tuple[float, float], rule: Tuple[float, float], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return discretized bins indices for a rule, considering only values in a range

    :param value_range: value range for bins
    :param rule: rule with min and max value for a feature
    :param n: number of bins to generate
    :return: tuple of bins index for the min and max value of the rule, and relative mean feature value
    """
    bins = np.linspace(start=value_range[0], stop=value_range[1], num=n + 1, endpoint=True)
    binned_rule = np.clip(np.digitize(rule, bins) - 1, a_min=0, a_max=n - 1)  # (2, )
    # Original values from digitize are in [1, n+1]
    # We normalize them in [0, n)

    # Average RGB color in each consecutive interval
    indices_pairs = np.array([[i, i + 1] for i in range(bins.shape[0] - 1)])
    colors = bins[indices_pairs].mean(axis=-1)  # n colors, 1 for each bin
    return binned_rule, colors


def create_plot(n_voxels, face_colors) -> Axes:
    """
    Create a voxel plot with specified cubes and colors

    :param n_voxels: bool 3D array with cubes to be filled with color
    :param face_colors: 4D RGBA colors for each cube
    :return: matplotlib axes of the produced plot
    """
    cubes_n = n_voxels.shape[0]
    filled = np.ones(n_voxels.shape)
    # Upscale the above voxel image, leaving gaps
    filled = explode(filled)
    face_colors = explode(face_colors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    ax = plt.figure(figsize=(10, 10), dpi=200).add_subplot(projection="3d")
    ax.voxels(x, y, z, filled, facecolors=face_colors, edgecolors=face_colors)
    ax.set_aspect("equal")
    ax.axes.set_xlim3d(left=0.000001, right=cubes_n + 0.9999999)
    ax.axes.set_ylim3d(bottom=0.000001, top=cubes_n + 0.9999999)
    ax.axes.set_zlim3d(bottom=0.000001, top=cubes_n + 0.9999999)
    ax.set_xlabel("R", fontsize=18)
    ax.set_ylabel("G", fontsize=18)
    ax.set_zlabel("B", fontsize=18)
    return ax


def plot_rgb_explanation(rule: List[Tuple[float, float]], cubes_n: int, alpha: float,
                         r_range: Tuple[float, float], g_range: Tuple[float, float], b_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce voxel plot "filled" and "facecolors" parameters for a single decision rule.

    :param rule: rule extracted from DT, a tuple of min and max value for each feature
    :param cubes_n: number of cubes to plot (tradeoff for speed, 256 is too much)
    :param alpha: value for alpha channel of all cubes
    :param r_range: range to show for R channel
    :param g_range: range to show for G channel
    :param b_range: range to show for B channel
    """

    r_rule, r_colors = create_bins_range(value_range=r_range, rule=rule[0], n=cubes_n)
    g_rule, g_colors = create_bins_range(value_range=g_range, rule=rule[1], n=cubes_n)
    b_rule, b_colors = create_bins_range(value_range=b_range, rule=rule[2], n=cubes_n)
    colors = np.stack([r_colors, g_colors, b_colors], axis=1)  # (cubes_n, 3)

    # bins = np.linspace(start=0.0, stop=1.0, num=cubes_n + 1, endpoint=True)
    # np_rule = np.clip(np.digitize(rule, bins) - 1, a_min=0, a_max=cubes_n - 1)  # (feature_n, 2), values in [0, max_value)
    # Average RGB color in each consecutive interval
    # indices = np.array([[i, i + 1] for i in range(bins.shape[0] - 1)])
    # colors = bins[indices].mean(axis=-1)  # max_value colors
    # del bins
    # del indices

    # Remove duplicated RGB tuples
    # np_rules = np.unique(np_rules, axis=0)

    # Expanded set of rules
    r_values = np.arange(start=r_rule[0], stop=r_rule[1] + 1, dtype=int)
    g_values = np.arange(start=g_rule[0], stop=g_rule[1] + 1, dtype=int)
    b_values = np.arange(start=b_rule[0], stop=b_rule[1] + 1, dtype=int)
    # Compute all possible combinations of RGB allowed values
    combination = np.array(np.meshgrid(r_values, g_values, b_values)).T.reshape(-1, 3)
    np_rule = np.unique(combination, axis=0)  # (n_rules, 3)

    # Create voxels cubes
    n_voxels = np.zeros((cubes_n, cubes_n, cubes_n), dtype=bool)
    n_voxels[tuple(np_rule.T)] = True

    # Get R G B components for each rule, using bins indices from np_rule
    # Diagonal is indexing magic
    colors = colors[np_rule].diagonal(axis1=1, axis2=2)  # (n_rules, 3)

    facecolors = np.zeros((cubes_n, cubes_n, cubes_n, 4))
    # Add suitable alpha channel, 0 for black, .6 for colors (semi-transparent)
    colors = np.concatenate([colors, np.full((colors.shape[0], 1), alpha)], axis=1)  # (n_rules, 4)
    facecolors[tuple(np_rule.T)] = colors

    return n_voxels, facecolors


def interpret_decision_tree(decision_tree: DecisionTreeClassifier, feature_names: List, num_classes: int):
    rules_extracted: List[List[List[Tuple[float, float]]]] = get_leaf_constraints(decision_tree, feature_names=feature_names, num_classes=num_classes)

    if feature_names == ["u", "v"]:
        raise ValueError("Explanation with YUV unsupported at the moment.")
        # rules_extracted = [[[yuv2rgb(()) for (min_f, max_f) in r] for r in regions] for regions in rules_extracted]

    return rules_extracted


def main():
    # Range of colors to plot
    r_range = (0.41, 0.65)
    g_range = (0.35, 0.70)
    b_range = (0.13, 0.40)
    # Number of cubes for each axis. Each cube represent a different color.
    cube_n = 24
    # Transparency of the cubes. 0 is completely transparent, 1 is opaque.
    alpha_channel = .3

    dt, feature_names, classes = train_dt()
    rules: List[List[List[Tuple[float, float]]]] = interpret_decision_tree(dt, feature_names, classes)
    for ripeness, class_rules in enumerate(rules):
        voxels, face_colors = np.zeros((cube_n, cube_n, cube_n), dtype=bool), np.zeros((cube_n, cube_n, cube_n, 4), dtype=float)
        # Accumulate voxels representation from each leaf in a single vector
        for leaf_rule in class_rules:
            voxels_leaf, face_colors_leaf = plot_rgb_explanation(leaf_rule, cubes_n=cube_n, r_range=r_range, g_range=g_range, b_range=b_range, alpha=alpha_channel)
            # Union of cubes from different leaves
            voxels |= voxels_leaf
            assert np.all(face_colors.shape == face_colors_leaf.shape), "Wrong shapes to unite"
            face_colors = np.where(face_colors > 0, face_colors, face_colors_leaf)
        fig = create_plot(voxels, face_colors)
        fig.set_title(f"RGB area for ripeness value {ripeness}", fontsize=18)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{ripeness}_rgb.png")
        fig.clear()
        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()
