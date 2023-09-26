# Read data
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.bayesian_model.bayesian_ridge_classifier import RidgePriorClassifier
from src.nlp.dataset import train_val_test, compute_metrics
from src.nlp.simple_model.text_features import TextFeatureExtractor

classifier_types = [MultinomialNB, LinearSVC, LogisticRegression,
                    DecisionTreeClassifier, RidgePriorClassifier]


def main():
    for classifier_type in classifier_types:
        # ---------------------------------------
        train_config: dict = load_yaml("src/nlp/params/experiment.yml")
        clf = classifier_type()
        # ---------------------------------------
        print("*** Misogyny task")
        data = train_val_test(target="M")
        X, y = data["train"]["x"], data["train"]["y"]
        # ---------------------------------------
        print("------ Training")
        max_features = train_config["max_features"]
        fex = TextFeatureExtractor()
        bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                         ngram_range=(1, 3),
                                         max_features=max_features,
                                         token_pattern=None)
        # Step 1: Vectorization
        X_transformed = bow_vectorizer.fit_transform(X)
        with open('src/nlp/params/vocab.txt', 'w', encoding='utf-8') as f:
            for item in list(bow_vectorizer.vocabulary_.keys()):
                f.write("%s\n" % item)
        X_dense = X_transformed.toarray()
        # ---------------------------------------
        params = train_config["grid_search_params"][classifier_type.__name__]
        gs = GridSearchCV(clf, param_grid=params, verbose=1, refit=True, n_jobs=8)

        gs.fit(X_dense, y)
        # ---------------------------------------
        pprint(gs.best_params_)
        print("------ Testing")
        # Predicting with a test dataset
        predicted = gs.predict(bow_vectorizer.transform(data["test"]["x"]).toarray())

        m_f1 = compute_metrics(predicted, data["test"]["y"], classifier_type.__name__)["f1"]
        print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()
