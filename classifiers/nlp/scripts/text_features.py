import json
import re
from collections import defaultdict
import spacy
from spacy.lang.it import STOP_WORDS

punctuation = r"""!"'()*+,-./:;<=>?[\]^_`{|}~"""  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

import html.entities


def char_to_unicode(char):
    return "U+" + format(ord(char), '04X')


def separate_html_entities(text) -> str:
    # Regular expression pattern for HTML entities and emojis
    pattern = r"(&#x[\da-fA-F]+;|&#\d+;)"

    # Add spaces around matches
    modified_text = re.sub(pattern, r" \1 ", text)

    # Remove potential extra spaces
    clean_text = " ".join(modified_text.split())

    return clean_text


def replace_with_unicode(text, mapping: dict):
    # Regular expression pattern for HTML entities and emojis
    pattern = r'(&#x[\da-fA-F]+;|&#\d+;)'

    # Find all matches
    matches = re.findall(pattern, text)

    for match in matches:
        # Unescape HTML entity and convert to Unicode
        unicode = char_to_unicode(html.unescape(match))
        try:
            code = mapping[unicode]
        except KeyError:
            print(f"Can't find {unicode}")
            pass
        else:
            # Replace match with Unicode in text
            text = text.replace(match, code)

    return text


class TextFeatureExtractor:
    def __init__(self):
        self.__spacy_model = spacy.load("it_core_news_sm")
        with open("classifiers/nlp/full-emoji-list.json", mode="r") as f:
            emap = json.load(f)
        emap = [a for v in emap.values() for a in v]

        match = re.compile(r"[^\w]")
        self.__entity_map = {e["code"]: f":{match.sub('', e['description']).upper().strip()}:" for e in emap}

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
        # string_clean = re.sub("<MENTION_.>", "", string, flags=re.RegexFlag.IGNORECASE).strip()

        # Separate EMOJIS from adjacent words if necessary
        string_clean = separate_html_entities(string)
        # Replace EMOJIS and ENTITIES with codes like ":CODE:"
        string_clean = replace_with_unicode(string_clean, self.__entity_map)
        # Remove all substrings with < "anything but spaces" >
        string_clean = re.sub("<\S+>", "", string_clean, flags=re.RegexFlag.IGNORECASE).strip()
        # Remove double spaces
        string_clean = re.sub(" +", " ", string_clean)

        # Regular expression pattern with negative lookahead (remove all characters that are not A-z, 0-9, _, !, ? and all strings made of ":A-Z:"
        string_clean = re.sub(r"(?!:[A-Z]+:)[^\w\s!?]", "", string_clean)
        # string_clean = re.sub(r"[^a-zA-Z0-9!?' ]", "", string_clean)

        doc = self.__spacy_model(string_clean)
        # print(doc.text)
        # entities = [(i, i.label_, i.label) for i in doc.ents]
        # Lemmatizing each token and converting each token into lowercase
        # tokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc if not token.is_stop]
        tokens = [token.text.lower().strip() for token in doc if not token.is_stop]
        tokens = [t for t in tokens if t and (t.isalnum() and len(t) > 2)]
        # print(tokens)
        # pos_token = [token.pos_ for token in doc if not token.is_stop]

        # Removing stop words
        # tokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

        # token.dep_
        # tagged_token = f"{token.pos_}_{token.lemma_}"
        # print(token.text, token.pos_, token.dep_, token.tag_)

        return tokens
