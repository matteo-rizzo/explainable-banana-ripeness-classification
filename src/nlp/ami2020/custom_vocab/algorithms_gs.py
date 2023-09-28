# Read data
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.ami2020.dataset import train_val_test, compute_metrics
from src.nlp.ami2020.text_features import TextFeatureExtractor

classifier_types = [RidgeClassifier]


def main():
    for classifier_type in classifier_types:
        # ---------------------------------------
        train_config: dict = load_yaml("src/nlp/params/experiment.yml")
        synthetic_add: bool = train_config["add_synthetic"]
        clf = classifier_type()
        # ---------------------------------------
        print("*** Misogyny task")
        data = train_val_test(target="M", add_synthetic_train=synthetic_add)
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
