import copy
from typing import List, Tuple, Dict, Any

from sklearn import tree


def merge_rules(per_class_rules: List[List[List[Tuple[float, float]]]], new_rules: Dict[str, List[Tuple[float, bool]]],
                class_idx: int, max_val: float, min_val: float, feature_names: List[Any]) -> None:
    to_update: List[List[Tuple[float, float]]] = per_class_rules[class_idx]

    # We'll store the min and max threshold for each feature and save it in this dictionary
    condensed_rule: Dict[str, Tuple[float, bool]] = dict()
    for feature_name, path in new_rules.items():
        node_min = min_val
        node_max = max_val
        for (node_threshold, sign) in path:
            if sign is False:
                # <=
                if node_threshold < node_max:
                    node_max = node_threshold
            else:
                # >
                if node_threshold > node_min:
                    node_min = node_threshold
        condensed_rule[feature_name] = (node_min, node_max)
    assert len(condensed_rule.keys()) == len(feature_names), "Wrong size"
    # Ensure features are in desired order, as explicit in "feature_names"
    feature_ranges: List[Tuple[float, float]] = [condensed_rule[fn] for fn in feature_names]
    to_update.append(feature_ranges)


def get_leaf_constraints(clf: tree.DecisionTreeClassifier, feature_names: List[Any], num_classes: int) -> List[
    List[List[Tuple[float, float]]]]:
    """
    For each target class, return the set of constraints on each feature that lead to that classification

    :param clf: decision tree classifier (DT)
    :param num_classes: number of predicted classes in the DT
    :param feature_names: list of names to use for each feature in the DT, with their position matching the position in the DT
    :return: for each output class, a list of feature ranges that lead to that prediction (as tuple of feature values)
    """
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    prediction = clf.tree_.value.argmax(axis=2).reshape(-1)

    # For each class, we want a list of leaf rules leading to that class
    # Each leaf rule is a List of ranges (min-max) for each feature
    per_class_rules: List[List[List[Tuple[float, float]]]] = [[] for _ in range(num_classes)]
    rules: Dict[str, List[Tuple[float, bool]]] = {fn: list() for fn in feature_names}

    stack = [(0, rules)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, rule_path = stack.pop()

        node_feature, node_threshold = feature[node_id], threshold[node_id]

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            # Decision node

            ln = copy.deepcopy(rule_path)
            rn = copy.deepcopy(rule_path)

            ln[feature_names[node_feature]].append((node_threshold, False))  # <=
            rn[feature_names[node_feature]].append((node_threshold, True))  # >

            # Append left and right children and depth to `stack`
            stack.append((children_left[node_id], ln))
            stack.append((children_right[node_id], rn))
        else:
            # Leaf node
            if feature_names == ["u", "v"]:
                # U and V values should be in [-.5, .5]
                max_val = .5
                min_val = -.5
            else:
                # RGB in [0, 1]
                max_val = 1.0
                min_val = .0
            merge_rules(per_class_rules, rule_path, prediction[node_id], max_val=max_val, min_val=min_val,
                        feature_names=feature_names)

    assert sum([len(a) for a in per_class_rules]) == clf.tree_.n_leaves, "Wrong number of leaves and outputs"
    return per_class_rules
