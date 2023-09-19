import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from classifiers.nlp.scripts.extract_features import train_val_test
from classifiers.nlp.scripts.text_features import TextFeatureExtractor
import matplotlib.patches as mpatches

# Ignore all warnings
warnings.filterwarnings('ignore')


def plot_coeffs(df: pd.DataFrame,
                title: str = "default_title",
                folder: Path = Path(""),
                n_extremes: int = 10,
                log_scale: bool = False):
    # -----------------------------------------
    # Get top 10 features
    top_10 = df.nlargest(n_extremes, 'Coefficient')
    # Get bottom 10 features
    bottom_10 = df.nsmallest(n_extremes, 'Coefficient')
    # Concatenate top and bottom 10
    df_top_bottom = pd.concat([top_10, bottom_10])
    # -----------------------------------------
    # Set dark grid background
    sns.set_style("darkgrid")

    # Create a bar plot of the coefficients
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_top_bottom['Feature'], df_top_bottom['Coefficient'])

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
        plt.title(f"{title} (log scale)")
    else:
        plt.title(title)

    plt.savefig(folder / f'{title.lower().replace(" ", "_")}.png')
    # -----------------------------------------


def naive_explained_classifier(target: str = "M"):
    # Read data
    data = train_val_test(target=target)

    classifier = LogisticRegression()

    fex = TextFeatureExtractor()
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer, ngram_range=(1, 3), max_features=10000)

    # Create a pipeline using TF-IDF
    pipe = Pipeline([('vectorizer', bow_vectorizer),
                     ('classifier', classifier)])

    print(f"Training on {'aggressiveness' if target == 'A' else 'misogyny'}...")

    pipe.fit(data["train"]["x"], data["train"]["y"])
    # --------------------------------------------------------------------
    # Get feature names from TfidfVectorizer
    feature_names = pipe.named_steps['vectorizer'].get_feature_names_out()

    # Get coefficients from the model
    coefficients = pipe.named_steps['classifier'].coef_[0]

    # Create a DataFrame for easy visualization
    df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    df = df.sort_values(by='Coefficient', ascending=True)
    # Save DataFrame to a CSV file
    coeff_folder = Path("results/coefficients")
    coeff_folder.mkdir(exist_ok=True, parents=True)
    # ---------
    # raw coefficients from logistic regression (centered around 0 and log scale)
    df.to_csv(coeff_folder / f'{target}_coefficients_log_odds.csv', index=False)
    plot_coeffs(df,
                f"{'Aggressiveness' if target == 'A' else 'Misogyny'} Log Odds Coefficients",
                coeff_folder)
    # ---------
    # exponent-ed log-odds (centered around 1, linear scale)
    df['Coefficient'] = np.exp(df['Coefficient'])
    df.to_csv(coeff_folder / f'{target}_coefficients_odds_ratio.csv', index=False)
    plot_coeffs(df,
                f"{'Aggressiveness' if target == 'A' else 'Misogyny'} Odds Ratio Coefficients",
                coeff_folder,
                log_scale=True)
    # --------------------------------------------------------------------
    print(f"Testing {'aggressiveness' if target == 'A' else 'Misogyny'}...")

    # Predicting with a test dataset
    predicted = pipe.predict(data["test"]["x"])

    # Model Accuracy
    classifier_name = classifier.__class__.__name__
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(data["test"]["y"], predicted,
                                                                             average="binary", pos_label=1)
    print(f"{classifier_name} accuracy:", metrics.accuracy_score(data["test"]["y"], predicted))
    print(f"{classifier_name} precision:", precision)
    print(f"{classifier_name} recall:", recall)
    print(f"{classifier_name} F1-score:", f1_score)


if __name__ == '__main__':
    naive_explained_classifier("A")
    naive_explained_classifier("M")
