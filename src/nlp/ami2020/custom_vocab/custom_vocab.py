import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.nlp.ami2020.text_features import TextFeatureExtractor


def make_custom_vectorizer(X, y, **kwargs):
    # * 1 --- Initial vocabulary calculation ---
    initial_vectorizer = TfidfVectorizer(**kwargs)
    initial_vectorizer.fit_transform(raw_documents=X, y=y)
    initial_vocabulary = initial_vectorizer.get_feature_names_out()
    # * 2 --- Addition/removal of words ---
    with open("dataset/AMI2020/lexicon/common_words.txt", 'r') as f:
        common_words = set(word.strip() for word in f)
    with open("dataset/AMI2020/lexicon/bad_words.txt", 'r') as f:
        bad_words = set(word.strip() for word in f)

    modified_vocabulary = []
    for term in initial_vocabulary:
        words = term.split()
        if any(word in bad_words for word in words):
            modified_vocabulary.append(term)
        elif all(word in common_words for word in words):
            continue
        else:
            modified_vocabulary.append(term)

    # * 3 --- Creation of new Vectorizer ---
    kwargs["max_features"] = len(modified_vocabulary)
    modified_vectorizer = TfidfVectorizer(**kwargs, vocabulary=modified_vocabulary)
    return modified_vectorizer


def fit_custom_pipeline(sk_classifier, X, y, max_features: int):
    fex = TextFeatureExtractor()
    # bow_vectorizer = make_custom_vectorizer(X, y,
    #                                         tokenizer=fex.preprocessing_tokenizer,
    #                                         ngram_range=(1, 3),
    #                                         max_features=max_features,
    #                                         token_pattern=None)
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=max_features,
                                     token_pattern=None)

    # Create a pipeline using TF-IDF
    # Step 1: Vectorization
    X_transformed = bow_vectorizer.fit_transform(X)

    # Step 2: Conversion to dense array
    X_dense = X_transformed.toarray()

    # Step 3: Classification
    sk_classifier.fit(X_dense, y)

    return sk_classifier, bow_vectorizer


def custom_vocab_classifier(sk_classifier, training_data: dict[str, dict[str, list]],
                            max_features: int = 10000) -> np.ndarray | tuple[np.ndarray, Pipeline]:
    print("------ Training")
    clf, vect = fit_custom_pipeline(sk_classifier, training_data["train"]["x"],
                                    training_data["train"]["y"], max_features)

    print("------ Testing")

    # Predicting with a test dataset
    predicted = clf.predict(vect.transform(training_data["test"]["x"]).toarray())
    return predicted, clf, vect
