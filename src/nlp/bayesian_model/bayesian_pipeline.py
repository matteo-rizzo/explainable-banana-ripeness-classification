import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.nlp.simple_model.text_features import TextFeatureExtractor


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()


def fit_pipeline(sk_classifier: ClassifierMixin, X, y):
    fex = TextFeatureExtractor()
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=10000,
                                     token_pattern=None)

    # Create a pipeline using TF-IDF
    # Step 1: Vectorization
    X_transformed = bow_vectorizer.fit_transform(X)

    # Step 2: Conversion to dense array
    X_dense = X_transformed.toarray()

    # Step 3: Classification
    # Neutral prior (no knowledge)
    prior = np.zeros(10000)
    sk_classifier.fit_with_prior(X_dense, y, prior=prior)

    return sk_classifier, bow_vectorizer


def make_pipeline(sk_classifier: ClassifierMixin) -> Pipeline:
    fex = TextFeatureExtractor()
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=10000,
                                     token_pattern=None)

    # Create a pipeline using TF-IDF
    pipe = Pipeline([('vectorizer', bow_vectorizer),
                     ('to_dense', DenseTransformer()),
                     ('classifier', sk_classifier)])
    return pipe


def bayesian_classifier(sk_classifier: ClassifierMixin, training_data: dict[str, dict[str, list]],
                        return_pipe: bool = False) -> np.ndarray | tuple[np.ndarray, Pipeline]:
    print("------ Training")
    clf, vect = fit_pipeline(sk_classifier, training_data["train"]["x"], training_data["train"]["y"])

    print("------ Testing")

    # Predicting with a test dataset
    predicted = clf.predict(vect.transform(training_data["test"]["x"]).toarray())

    if not return_pipe:
        return predicted
    else:
        return predicted, clf, vect
