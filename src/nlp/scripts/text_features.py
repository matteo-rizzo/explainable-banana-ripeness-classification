import json
import logging
import re
from collections import defaultdict
import spacy
from spacy import Language
from spacy.lang.it import STOP_WORDS
from treetaggerwrapper import TreeTagger

from src.classifiers.deep_learning.functional.yaml_manager import load_yaml

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
        experiment_config: dict = load_yaml("src/nlp/params/experiment.yml")

        global TREETAGGER
        if experiment_config["tree_tagger_path"]:
            logging.warning(" ********** TreeTagger option is selected. "
                            " ********** This requires installing 'treetaggerwrapper' and TreeTagger.\n"
                            " ********** See https://pypi.org/project/treetaggerwrapper for instructions. ")
            TREETAGGER = TreeTagger(TAGDIR=experiment_config["tree_tagger_path"], TAGLANG="it")
        else:
            TREETAGGER = None

        self.__spacy_model = spacy.load("it_core_news_lg")
        if TREETAGGER is not None:
            self.__spacy_model.replace_pipe("lemmatizer", "tree_tagger")
        with open("src/nlp/full-emoji-list.json", mode="r", encoding="utf-8") as f:
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

        # Regular expression pattern with negative lookahead (remove all characters that are not A-z, 0-9, _, !, ? and all strings made of ":A-Z:", removing the colons
        string_clean = re.sub(r"(?!:[A-Z]+:)[^\w\s!?]", "", string_clean)
        string_clean = re.sub(r":", "", string_clean).strip()

        string_empty = len(string_clean) == 0
        if string_empty:
            # Encode empty string with this string
            string_clean = "EMPTYSTRING"
        doc = self.__spacy_model(string_clean)
        # print(doc.text)
        # entities = [(i, i.label_, i.label) for i in doc.ents]
        # Lemmatizing each token and converting each token into lowercase
        # tokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc if not token.is_stop]
        tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop]
        tokens = [t for t in tokens if t and (not t.isalnum() or len(t) > 2)]
        # print(tokens)
        # pos_token = [token.pos_ for token in doc if not token.is_stop]

        # if string_empty: # debug
        #     print(tokens)

        # token.dep_
        # tagged_token = f"{token.pos_}_{token.lemma_}"
        # print(token.text, token.pos_, token.dep_, token.tag_)

        return tokens


@Language.component("tree_tagger")
def tree_tagger(doc):
    """ Tagger component of TreeTagger compatible with Spacy pipeline """
    tokens = [token.text for token in doc if not token.is_space]

    tags = TREETAGGER.tag_text(tokens, tagonly=True)
    lemmas = [tag.split("\t")[2].split("|")[0] for tag in tags]

    j = 0
    for token in doc:
        if not token.is_space:
            token.lemma_ = lemmas[j]
            j += 1
        else:
            token.lemma_ = " "

    return doc


if __name__ == '__main__':
    fex = TextFeatureExtractor()
    # s = "<MENTION_1> So che tu sei acida&#x1f602; a"
    # s = "Porca troia che gran pezzo di figa"
    s = "Ti sfonderei tutta&#x1f61c;&#x1f61c;as dc"
    rs = fex.preprocessing_tokenizer(s)
    pass
