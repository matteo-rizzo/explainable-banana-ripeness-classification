import re
from collections import defaultdict
import spacy
from spacy.lang.it import STOP_WORDS

punctuation = r"""!"'()*+,-./:;<=>?[\]^_`{|}~"""  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""


class TextFeatureExtractor:
    def __init__(self):
        self.__spacy_model = spacy.load("it_core_news_sm")

    def count_characters(self, s: str):
        d = defaultdict(int)
        for t in s:
            if t in punctuation:
                d["punc"] += 1
            elif t.isalpha():
                d['letters'] += 1
            elif t.isdigit():
                d['numbers'] += 1
            elif t.isspace():
                pass
            else:
                d['other'] += 1  # this will include spaces

            if t.isupper():
                d["uppercase"] += 1

        d["length"] = len(s)

        return d

    def preprocessing_tokenizer(self, string: str) -> list[str]:
        string_clean = re.sub("<MENTION_.>", "", string, flags=re.RegexFlag.IGNORECASE).strip()
        string_clean = re.sub(" +", " ", string_clean)
        string_clean = re.sub(r"[^a-zA-Z0-9!? ]", "", string_clean)

        doc = self.__spacy_model(string_clean)
        # print(doc.text)
        # entities = [(i, i.label_, i.label) for i in doc.ents]
        # Lemmatizing each token and converting each token into lowercase
        # tokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc if not token.is_stop]
        tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop]
        pos_token = [token.pos_ for token in doc if not token.is_stop]

        # Removing stop words
        # tokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

        # token.dep_
        # tagged_token = f"{token.pos_}_{token.lemma_}"
        # print(token.text, token.pos_, token.dep_, token.tag_)

        return tokens
