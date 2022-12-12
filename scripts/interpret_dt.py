from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import yuv2rgb
from sklearn.tree import DecisionTreeClassifier

from scripts.decision_tree import get_leaf_constraints, train_dt


def explode(data):
    size = np.array(data.shape) * 2
    if len(size) > 3:
        size = np.array([*size[:-1], data.shape[-1] + 1])
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def plot_rgb_explanation(rules: List[Tuple], max_values: int = None):
    """
    Produce Voxel plot

    :param rules: rules extracted from DT
    :param max_values: number of cubes to plot (tradeoff for speed, 256 is too much)
    """
    res_max = 256

    if max_values is not None:
        bins = np.linspace(start=0.0, stop=1.0, num=max_values + 1, endpoint=True)
        np_rules = np.clip(np.digitize(rules, bins) - 1, a_min=0, a_max=max_values - 1)  # (rules_n, 3), values in [0, max_value)
        res_max = max_values
        # Average RGB color in each consecutive interval
        indices = np.array([[i, i + 1] for i in range(bins.shape[0] - 1)])
        colors = bins[indices].mean(axis=-1)  # max_value colors
        del bins
        del indices
    else:
        int_rgbs = [tuple([int(c * (res_max - 1)) for c in components]) for components in rules]
        np_rules = np.array(int_rgbs)  # (rules_n, 3)

    # Remove duplicated RGB tuples
    np_rules = np.unique(np_rules, axis=0)

    # Expanded set of rules
    # TODO: check this and rule generation
    ext_rules = list()
    for rule in np_rules:
        # rule is a triple RGB
        r = np.arange(rule[0] + 1, dtype=int)
        g = np.arange(rule[1] + 1, dtype=int)
        b = np.arange(rule[2] + 1, dtype=int)
        # Compute all possible combinations of RGB allowed values
        combination = np.array(np.meshgrid(r, g, b)).T.reshape(-1, 3)
        ext_rules.append(combination)
    ext_rules = np.concatenate(ext_rules, axis=0)
    np_rules = np.unique(ext_rules, axis=0)

    # create voxels cubes
    n_voxels = np.zeros((res_max, res_max, res_max), dtype=bool)
    n_voxels[tuple(np_rules.T)] = True

    if max_values is None:
        colors = np_rules
    else:
        colors = colors[np_rules]

    facecolors = np.zeros((res_max, res_max, res_max, 4))
    # Add suitable alpha channel, 0 for black, .6 for colors (semi-tranparent)
    colors = np.concatenate([colors, np.full((colors.shape[0], 1), .6)], axis=1)  # (n_rules, 4)
    facecolors[tuple(np_rules.T)] = colors

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

    ax = plt.figure(figsize=(15, 15), dpi=300).add_subplot(projection="3d")
    ax.voxels(x, y, z, filled, facecolors=facecolors, edgecolors=facecolors)
    ax.set_aspect("equal")

    plt.show()


def interpret_decision_tree(decision_tree: DecisionTreeClassifier, feature_names: List, num_classes: int):
    rules_extracted: list[list[tuple[float]]] = get_leaf_constraints(decision_tree, feature_names=feature_names, num_classes=num_classes)

    if feature_names == ["u", "v"]:
        rules_extracted = [[tuple((yuv2rgb((.65, *r)) * 255).round().astype(int)) for r in regions] for regions in rules_extracted]

    # plot_hull_section_colours(hull, section_colours="RGB", section_opacity=1.0)
    # cat_1 = np.array(rules_extracted[0])
    # opacity = np.array([1.0] * len(cat_1)).reshape(-1, 1)

    # plot_RGB_colourspace_section(
    #     "sRGB", section_colours=cat_1, section_opacity=opacity
    # )

    return rules_extracted


def main():
    dt, feature_names, classes = train_dt()
    rules = interpret_decision_tree(dt, feature_names, classes)
    plot_rgb_explanation(rules[1], max_values=16)


if __name__ == "__main__":
    main()
