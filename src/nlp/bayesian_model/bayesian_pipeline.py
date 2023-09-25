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


class CustomClassifier(ClassifierMixin):
    def __init__(self, classifier, prior=None):
        self.classifier = classifier
        self.prior = prior

    def fit(self, X, y):
        if self.prior is not None:
            new_y = y - np.sum(self.prior * X, axis=1)
            self.classifier.fit(X, new_y)
            self.classifier.coef_ += self.prior
        else:
            self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)


def bayesian_make_pipeline(sk_classifier: ClassifierMixin, prior=None) -> Pipeline:
    fex = TextFeatureExtractor()
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=10000,
                                     token_pattern=None)

    # Create a pipeline using TF-IDF
    pipe = Pipeline([('vectorizer', bow_vectorizer),
                     ('to_dense', DenseTransformer()),
                     ('classifier', CustomClassifier(sk_classifier, prior))])
    return pipe


def fit_pipeline(sk_classifier, X, y):
    max_features = 8000
    fex = TextFeatureExtractor()
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=max_features,
                                     token_pattern=None)

    # Create a pipeline using TF-IDF
    # Step 1: Vectorization
    X_transformed = bow_vectorizer.fit_transform(X)

    with open('src/nlp/params/vocab.txt', 'w', encoding='utf-8') as f:
        for item in list(bow_vectorizer.vocabulary_.keys()):
            f.write("%s\n" % item)

    with open('src/nlp/params/test_prior.txt', 'r', encoding='utf-8') as f:
        prior_list = {line.split()[0]: float(line.split()[1]) for line in f}

    # Step 2: Conversion to dense array
    X_dense = X_transformed.toarray()
    # X_dense = np.where(X_dense > 0, 1, X_dense)
    # Step 3: Classification
    # Neutral prior (no knowledge)
    prior = np.zeros(max_features)

    for word, value in prior_list.items():
        for vocab_word, index in bow_vectorizer.vocabulary_.items():
            if word in vocab_word:
                prior[index] += value

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


def bayesian_classifier(sk_classifier, training_data: dict[str, dict[str, list]],
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
