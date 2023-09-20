import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeClassifier

from classifiers.nlp.scripts.pipeline import train_val_test, naive_classifier, compute_metrics
from classifiers.nlp.scripts.run_classification import classifier_type
from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml

# Ignore all warnings
warnings.filterwarnings('ignore')

classifier_type = RidgeClassifier


def plot_global_coeffs(df: pd.DataFrame, title: str = "default_title",
                       folder: Path = Path(""), n_extremes: int = 10,
                       log_scale: bool = False):
    # -----------------------------------------
    # Get top features
    top = df.nlargest(n_extremes, 'Coefficient')
    # Get bottom features
    bottom = df.nsmallest(n_extremes, 'Coefficient')
    # Concatenate top and bottom
    df_top_bottom = pd.concat([top, bottom])
    # -----------------------------------------
    # Set dark grid background
    sns.set_style("darkgrid")

    # Create a bar plot of the coefficients
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_top_bottom['Feature'], df_top_bottom['Coefficient'])
    n_extremes = min(n_extremes, len(bars))
    # Color the top bars in red
    for i in range(n_extremes):
        bars[i].set_color('blue')
        bars[-(i + 1)].set_color('red')

    # Create custom patches for the legend
    red_patch = mpatches.Patch(color='red', label='Negative Contribution')
    blue_patch = mpatches.Patch(color='blue', label='Positive Contribution')

    # Add the legend with custom patches
    plt.legend(handles=[red_patch, blue_patch])

    plt.xlabel('Coefficient')
    plt.ylabel('Feature')

    if log_scale:
        # Add logarithmic scale to x-axis
        plt.xscale('log')
        plt.title(f"{title} (log scale) (top/bottom {n_extremes})")
    else:
        plt.title(f"{title} (top/bottom {n_extremes})")

    plt.savefig(folder / f'{title.lower().replace(" ", "_")}.png')
    # -----------------------------------------


def plot_local_coeffs(df: pd.DataFrame, title: str = "default_title",
                      folder: Path = Path(""), max_values: int = 10, msg=""):
    # Set dark grid background
    sns.set_style("darkgrid")
    # Keep most impactful
    if len(df) > max_values:
        df = df.nlargest(max_values, 'Coefficient')
    # Create a subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create a bar plot of the coefficients
    bars = ax.barh(df['Feature'], df['Coefficient'])
    # Color the bars and add labels
    for bar in bars:
        # Width is the value
        if bar.get_width() > 0:
            bar.set_color('blue')
        else:
            bar.set_color('red')
    ax.bar_label(bars, fmt='%.2f')

    # Create custom patches for the legend
    red_patch = mpatches.Patch(color='red', label='Negative Contribution')
    blue_patch = mpatches.Patch(color='blue', label='Positive Contribution')

    # Add the legend with custom patches
    plt.legend(handles=[red_patch, blue_patch])

    plt.xlabel('Coefficient')
    plt.ylabel('Feature')

    # Add logarithmic scale to x-axis
    fig.suptitle(f"{title}")
    ax.title.set_text(msg)

    plt.savefig(folder / f'{title.lower().replace(" ", "_")}.png')
    # -----------------------------------------


def naive_explained_classifier(target: str = "M"):
    train_config: dict = load_yaml("params/experiment.yml")
    clf_params = train_config[classifier_type.__name__]

    print(f"Training {'Aggressiveness' if target == 'A' else 'Misogyny'}...")
    data = train_val_test(target=target)
    pred, pipe = naive_classifier(classifier_type(**clf_params), data, return_pipe=True)
    classifier = pipe.named_steps['classifier']

    # --------------------------------------------------------------------
    # Get feature names from vectorizer
    feature_names = pipe.named_steps['vectorizer'].get_feature_names_out()

    # Get coefficients from the model
    coefficients = classifier.coef_[0]

    # Create a DataFrame for easy visualization
    df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    df = df.sort_values(by='Coefficient', ascending=False)
    # Save DataFrame to a CSV file
    coeff_folder = Path("results/coefficients")
    coeff_folder.mkdir(exist_ok=True, parents=True)
    # ---------
    # raw coefficients from logistic regression (centered around 0 and log scale)
    df.to_csv(coeff_folder / f'{target}_coefficients_log_odds.csv', index=False)
    plot_global_coeffs(df,
                       f"{'Aggressiveness' if target == 'A' else 'Misogyny'} Log Odds Coefficients",
                       coeff_folder,
                       log_scale=False)
    # ---------
    # # exponent-ed log-odds (centered around 1, linear scale)
    # df['Coefficient'] = np.exp(df['Coefficient'])
    # df.to_csv(coeff_folder / f'{target}_coefficients_odds_ratio.csv', index=False)
    # plot_global_coeffs(df,
    #                    f"{'Aggressiveness' if target == 'A' else 'Misogyny'} Odds Ratio Coefficients",
    #                    coeff_folder,
    #                    log_scale=True)
    # --------------------------------------------------------------------
    print(f"Testing {'Aggressiveness' if target == 'A' else 'Misogyny'}...")
    compute_metrics(pred, data["test"]["y"], classifier_type.__name__)

    return pipe.named_steps['vectorizer'], pipe.named_steps['classifier']


def local_prediction_features(doc: str, vectorizer, model):
    # Transform the document into TF-IDF features
    X = vectorizer.transform([doc])

    # Get the names of the features from TfidfVectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Get the non-zero features for this document
    nz_features = X.nonzero()[1]

    # Get the corresponding coefficients
    coeffs = model.coef_[0, nz_features]

    # Create a DataFrame for easy visualization
    df = pd.DataFrame({'Feature': np.array(feature_names)[nz_features], 'Coefficient': coeffs})

    # Sort DataFrame by Coefficient value
    df = df.sort_values(by='Coefficient', ascending=False)

    return df


def sample_examples(vectorizer, classifier, clf_type: str):
    # -------------------------------------------------------
    coeff_folder = Path("results/coefficients/examples")
    coeff_folder.mkdir(parents=True, exist_ok=True)
    # -------------------------------------------------------
    test_set = pd.read_table("dataset/AMI2020/testset/AMI2020_test_raw_gold_anon.tsv")
    neither = test_set[(test_set["misogynous"] == 0) & (test_set["aggressiveness"] == 0)]
    misogynous = test_set[(test_set["misogynous"] == 1) & (test_set["aggressiveness"] == 0)]
    both = test_set[(test_set["misogynous"] == 1) & (test_set["aggressiveness"] == 1)]
    # -------------------------------------------------------
    neither_samples = neither.sample(2)["text"]
    misogynous_samples = misogynous.sample(2)["text"]
    both_samples = both.sample(2)["text"]
    # -------------------------------------------------------
    for idx, neither_sample in neither_samples.iteritems():
        df = local_prediction_features(neither_sample, vectorizer, classifier)
        plot_local_coeffs(df,
                          f"{idx} {clf_type} Log Odds Coefficients 0 0",
                          coeff_folder, msg=neither_sample)
    # -------------------------------------------------------
    for idx, misogynous_sample in misogynous_samples.iteritems():
        df = local_prediction_features(misogynous_sample, vectorizer, classifier)
        plot_local_coeffs(df,
                          f"{idx} {clf_type} Log Odds Coefficients 1 0",
                          coeff_folder, msg=misogynous_sample)
    # -------------------------------------------------------
    for idx, both_sample in both_samples.iteritems():
        df = local_prediction_features(both_sample, vectorizer, classifier)
        plot_local_coeffs(df,
                          f"{idx} {clf_type} Log Odds Coefficients 1 1",
                          coeff_folder, msg=both_sample)
    # -------------------------------------------------------


if __name__ == '__main__':
    a_vectorizer, a_classifier = naive_explained_classifier("A")
    m_vectorizer, m_classifier = naive_explained_classifier("M")

    sample_examples(a_vectorizer, a_classifier, "Aggressiveness")
    sample_examples(m_vectorizer, m_classifier, "Mysogyny")
