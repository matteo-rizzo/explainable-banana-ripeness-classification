from typing import List, Tuple, Optional

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


def plot_rgb_explanation(rule: List[Tuple[float, float]], max_plot_components: int = None, axes: Axes = None, title: Optional[str] = None) -> Axes:
    """
    Produce Voxel plot

    :param rule: rule extracted from DT, a tuple of min and max value for each feature
    :param max_plot_components: number of cubes to plot (tradeoff for speed, 256 is too much)
    """
    res_max = 256

    if max_plot_components is not None:
        bins = np.linspace(start=0.0, stop=1.0, num=max_plot_components + 1, endpoint=True)
        np_rule = np.clip(np.digitize(rule, bins) - 1, a_min=0, a_max=max_plot_components - 1)  # (feature_n, 2), values in [0, max_value)
        res_max = max_plot_components
        # Average RGB color in each consecutive interval
        indices = np.array([[i, i + 1] for i in range(bins.shape[0] - 1)])
        colors = bins[indices].mean(axis=-1)  # max_value colors
        del bins
        del indices
    else:
        int_rgbs = [[(int(b * (res_max - 1)), int(t * (res_max - 1))) for (b, t) in components] for components in rule]
        np_rule = np.array(int_rgbs)  # (feature_n, 2)

    # Remove duplicated RGB tuples
    # np_rules = np.unique(np_rules, axis=0)

    # Expanded set of rules
    r = np.arange(start=np_rule[0, 0], stop=np_rule[0, 1] + 1, dtype=int)
    g = np.arange(start=np_rule[1, 0], stop=np_rule[1, 1] + 1, dtype=int)
    b = np.arange(start=np_rule[2, 0], stop=np_rule[2, 1] + 1, dtype=int)
    # Compute all possible combinations of RGB allowed values
    combination = np.array(np.meshgrid(r, g, b)).T.reshape(-1, 3)
    np_rule = np.unique(combination, axis=0)

    # create voxels cubes
    n_voxels = np.zeros((res_max, res_max, res_max), dtype=bool)
    n_voxels[tuple(np_rule.T)] = True

    if max_plot_components is None:
        colors = np_rule
    else:
        colors = colors[np_rule]

    facecolors = np.zeros((res_max, res_max, res_max, 4))
    # Add suitable alpha channel, 0 for black, .6 for colors (semi-tranparent)
    colors = np.concatenate([colors, np.full((colors.shape[0], 1), .6)], axis=1)  # (n_rules, 4)
    facecolors[tuple(np_rule.T)] = colors

    filled = np.ones(n_voxels.shape)
    # upscale the above voxel image, leaving gaps
    filled = explode(filled)
    facecolors = explode(facecolors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    # f = create_voxel_figure(x, "test", fcolors_2)
    # a = np.stack([x, y, z]).T
    # f = _create_voxel_mesh_figure(a, a, fcolors_2, "test")
    # f.show()

    ax = axes
    if ax is None:
        ax = plt.figure(figsize=(15, 15), dpi=300).add_subplot(projection="3d")
    ax.voxels(x, y, z, filled, facecolors=facecolors, edgecolors=facecolors)
    ax.set_aspect("equal")
    ax.axes.set_xlim3d(left=0.000001, right=max_plot_components + 0.9999999)
    ax.axes.set_ylim3d(bottom=0.000001, top=max_plot_components + 0.9999999)
    ax.axes.set_zlim3d(bottom=0.000001, top=max_plot_components + 0.9999999)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    if title:
        ax.set_title(title)
    return ax


def interpret_decision_tree(decision_tree: DecisionTreeClassifier, feature_names: List, num_classes: int):
    rules_extracted: List[List[List[Tuple[float, float]]]] = get_leaf_constraints(decision_tree, feature_names=feature_names, num_classes=num_classes)

    if feature_names == ["u", "v"]:
        raise ValueError("Explanation with YUV unsupported at the moment.")
        # rules_extracted = [[[yuv2rgb(()) for (min_f, max_f) in r] for r in regions] for regions in rules_extracted]

    return rules_extracted


def main():
    dt, feature_names, classes = train_dt()
    rules: List[List[List[Tuple[float, float]]]] = interpret_decision_tree(dt, feature_names, classes)
    for ripeness, class_rules in enumerate(rules):
        fig = None
        for leaf_rule in class_rules:
            fig = plot_rgb_explanation(leaf_rule, max_plot_components=16, axes=fig)
        fig.set_title(f"RGB area for ripeness value {ripeness}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
