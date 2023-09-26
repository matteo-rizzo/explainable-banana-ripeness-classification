# Read data

from sklearn.feature_extraction.text import TfidfVectorizer

from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.bayesian_model.bayesian_ridge_classifier import RidgePriorClassifier
from src.nlp.dataset import train_val_test, compute_metrics
from src.nlp.text_features import TextFeatureExtractor

classifier_type = RidgePriorClassifier


def main():
    # ---------------------------------------
    train_config: dict = load_yaml("src/nlp/params/experiment.yml")
    clf_params = train_config[classifier_type.__name__]
    clf = RidgePriorClassifier(**clf_params)
    # ---------------------------------------
    print("*** Misogyny task")
    data = train_val_test(target="M")
    X, y = data["train"]["x"], data["train"]["y"]
    # ---------------------------------------
    print("------ Training")
    max_features = 8000
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

    clf.fit(X_dense, y)
    # ---------------------------------------
    print("------ Testing")
    # Predicting with a test dataset
    predicted = clf.predict(bow_vectorizer.transform(data["test"]["x"]).toarray())

    m_f1 = compute_metrics(predicted, data["test"]["y"], classifier_type.__name__)["f1"]


if __name__ == "__main__":
    main()
