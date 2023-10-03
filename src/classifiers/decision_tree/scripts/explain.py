import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt, gridspec
from sklearn.tree import DecisionTreeClassifier

from src.classifiers.decision_tree.run_dt_experiments import train_dt
from src.classifiers.decision_tree.scripts.extract_rules import get_leaf_constraints


def explode(data):
    size = np.array(data.shape) * 2
    if len(size) > 3:
        size = np.array([*size[:-1], data.shape[-1] + 1])
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def create_bins_range(value_range: Tuple[float, float], rule: Tuple[float, float], n: int) -> Tuple[
    np.ndarray, np.ndarray]:
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


def binarize_color(color: Tuple[float, float, float],
                   ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], n: int):
    bins = [np.linspace(start=ranges[0][0], stop=ranges[0][1], num=n + 1, endpoint=True),  # r
            np.linspace(start=ranges[1][0], stop=ranges[1][1], num=n + 1, endpoint=True),  # g
            np.linspace(start=ranges[2][0], stop=ranges[2][1], num=n + 1, endpoint=True)]  # b
    return tuple([int(np.clip(np.digitize(color[i], bins[i]) - 1, a_min=0, a_max=n - 1)) for i in range(len(color))])


def create_plot(ax, n_voxels, face_colors, edge_colors):
    """
    Create a voxel plot with specified cubes and colors

    :param n_voxels: bool 3D array with cubes to be filled with color
    :param face_colors: 4D RGBA colors for each cube
    :param edge_colors: 4D RGBA colors for the edges of each cube
    :return: matplotlib axes of the produced plot
    """
    cubes_n = n_voxels.shape[0]
    filled = np.ones(n_voxels.shape)
    # Upscale the above voxel image, leaving gaps
    filled = explode(filled)
    face_colors = explode(face_colors)
    edge_colors = explode(edge_colors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    ax.voxels(x, y, z, filled, facecolors=face_colors, edgecolors=edge_colors)
    ax.set_aspect("equal")
    ax.axes.set_xlim3d(left=0.000001, right=cubes_n + 0.9999999)
    ax.axes.set_ylim3d(bottom=0.000001, top=cubes_n + 0.9999999)
    ax.axes.set_zlim3d(bottom=0.000001, top=cubes_n + 0.9999999)
    ax.set_xlabel("R", fontsize=12)
    ax.set_ylabel("G", fontsize=12)
    ax.set_zlabel("B", fontsize=12)

    return ax


def rotate_ax(ax, angle, azim, roll):
    # for angle in range(0, 360 * 4 + 1):
    # Normalize the angle to the range [-180, 180] for display
    angle_norm = (angle + 180) % 360 - 180

    # Cycle through a full rotation of elevation, then azimuth, roll, and all
    # if angle <= 360:
    elev = angle_norm
    # elif angle <= 360 * 2:
    #     azim = angle_norm
    # elif angle <= 360 * 3:
    #     roll = angle_norm
    # else:
    #     elev = azim = roll = angle_norm

    # Update the axis view and title
    ax.view_init(elev, azim, roll)


def plot_rgb_explanation(rule: List[Tuple[float, float]], cubes_n: int, alpha: float,
                         r_range: Tuple[float, float],
                         g_range: Tuple[float, float],
                         b_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
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
    # np_rule = np.clip(np.digitize(rule, bins) - 1, a_min=0, a_max=cubes_n - 1)
    # (feature_n, 2), values in [0, max_value)
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
    rules_extracted: List[List[List[Tuple[float, float]]]] = get_leaf_constraints(decision_tree,
                                                                                  feature_names=feature_names,
                                                                                  num_classes=num_classes)

    if feature_names == ["u", "v"]:
        raise ValueError("Explanation with YUV unsupported at the moment.")

    return rules_extracted


IMGS = [[0.79197663, 0.8625892, 0.47860467],
        [0.8111009, 0.84216195, 0.53676605],
        [0.85316503, 0.82352686, 0.5224761],
        [0.8699858, 0.81373686, 0.5289033]]


def main():
    # Range of colors to plot
    r_range = (0.68, 0.93)
    g_range = (0.74, 0.92)
    b_range = (0.40, 0.64)
    # Number of cubes for each axis. Each cube represent a different color.
    cube_n = 6
    # Transparency of the cubes. 0 is completely transparent, 1 is opaque.
    alpha_channel = .3
    # Set whether a cube should be highlighted, or None if not
    highlight_color = None  # es: (0.5, 0.6, 0.3)

    dt, feature_names, num_classes = train_dt()
    print("-----------------------------------------------------------")
    print("Interpreting DT...")
    start = time.perf_counter()
    rules: List[List[List[Tuple[float, float]]]] = interpret_decision_tree(dt, feature_names, num_classes)
    for ripeness, class_rules in enumerate(rules):
        highlight_color = IMGS[ripeness]
        start_partial = time.perf_counter()
        voxels, face_colors = np.zeros((cube_n, cube_n, cube_n), dtype=bool), np.zeros((cube_n, cube_n, cube_n, 4),
                                                                                       dtype=float)
        # Accumulate voxels representation from each leaf in a single vector
        for leaf_rule in class_rules:
            voxels_leaf, face_colors_leaf = plot_rgb_explanation(leaf_rule, cubes_n=cube_n, r_range=r_range,
                                                                 g_range=g_range, b_range=b_range, alpha=alpha_channel)
            # Union of cubes from different leaves
            voxels |= voxels_leaf
            assert np.all(face_colors.shape == face_colors_leaf.shape), "Wrong shapes to unite"
            face_colors = np.where(face_colors > 0, face_colors, face_colors_leaf)

        edge_colors = face_colors

        # HIGHLIGHT OF A SINGLE CUBE IN THE PLOT
        if highlight_color is not None:
            # Find bin for color tuple
            rh, gh, bh = binarize_color(color=highlight_color, ranges=(r_range, g_range, b_range), n=cube_n)
            edge_colors = np.copy(face_colors)
            # Find indices for colored cubes in 3D
            colored_indices = np.unique(np.argwhere(face_colors > 0)[:, :-1], axis=0)  # (N, 3)
            # Set edges of cube to red for that single cube
            edge_colors[rh, gh, bh] = np.array([1, 0, 0, 1], dtype=float)  # color of target cube
            # Set high face transparency for all other cubes
            tmp = face_colors[tuple(colored_indices.T)]
            tmp[:, -1] = 0.05
            face_colors[tuple(colored_indices.T)] = tmp
            face_colors[rh, gh, bh, -1] = 1.0  # alpha for FACE of target cube
            # Adjust edge transparency
            tmp = edge_colors[tuple(colored_indices.T)]
            tmp[:, -1] = 0.1
            edge_colors[tuple(colored_indices.T)] = tmp
            edge_colors[rh, gh, bh, -1] = 1.0  # alpha for EDGE of target cube

        gs = gridspec.GridSpec(1, 1)
        fig = plt.figure(figsize=(14, 10), dpi=150)
        ax_main = fig.add_subplot(gs[0, 0], projection="3d")

        create_plot(ax_main, voxels, face_colors, edge_colors)

        # rotate_ax(ax_sub_3, angle=90, azim=0, roll=90)
        fig.suptitle(f"RGB area for ripeness value {ripeness}", fontsize=18)
        plots_path: Path = Path("plots")
        plots_path.mkdir(exist_ok=True)

        plt.savefig(plots_path / f"{ripeness}_{cube_n}_{'no-hl' if highlight_color is None else 'yes-hl'}_rgb.png")
        plt.cla()
        fig.clear()
        plt.clf()
        plt.close()
        end_partial = time.perf_counter()
        print(f"Finished a figure in {(end_partial - start_partial):.3f}s.")
    end = time.perf_counter()
    print(f"\nFinished all figures in {(end - start):.3f}s.")


if __name__ == "__main__":
    main()
