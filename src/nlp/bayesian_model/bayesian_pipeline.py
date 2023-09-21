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
