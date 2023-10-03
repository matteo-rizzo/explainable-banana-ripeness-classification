from math import ceil, floor, sqrt
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.classifiers.decision_tree.run_dt_experiments import train_dt, prepare_features
from src.classifiers.deep_learning.functional.yaml_manager import load_yaml


def analyze_tree_errors():
    # Load configuration
    train_config: Dict = load_yaml("params/networks/experiment.yml")
    num_rand_states: int = train_config["num_seeds"]
    test_size: float = train_config["test_size"]

    x, y, _ = prepare_features(train_config)
    dt, _, _ = train_dt()

    wrong_predictions: List = []
    for rs in range(num_rand_states):
        # Prepare splits
        _, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rs)
        y_train, y_test = y_train["y"].tolist(), y_test["y"].tolist()

        y_pred = dt.predict(x_test).tolist()

        for true, pred, sample in zip(y_test, y_pred, x_test.to_numpy().tolist()):
            if true != pred:
                wrong_predictions.append({"true": true,
                                          "pred": pred,
                                          "color": sample})

    df = pd.DataFrame(wrong_predictions)
    return df


def plot_errors(df: pd.DataFrame):
    img_num: int = len(df)
    n_rows: int = ceil(sqrt(img_num))
    n_cols: int = floor(sqrt(img_num))
    print(n_rows, n_cols)

    _, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
    # Just to hide blocks
    white_blocks: int = (n_cols * n_rows) - img_num
    df = df.append([pd.DataFrame([[-1, -1, [1.0, 1.0, 1.0]]]
                                 , columns=["true", "pred", "color"])] * white_blocks,
                   ignore_index=True)
    for (true, pred, color), ax in zip(df.values, axs.flatten()):
        color_block = np.array([[color] * 25] * 25)
        if true != -1:
            ax.text(1, 13, f"PRED: {pred}", style='italic', )
            ax.text(1, 18, f"TRUE: {true}", style='italic', )
        ax.axis('off')
        ax.imshow(color_block)
    plt.axis('off')
    plt.show(bbox_inches='tight')


def main():
    df = analyze_tree_errors()
    plot_errors(df)


if __name__ == "__main__":
    main()
