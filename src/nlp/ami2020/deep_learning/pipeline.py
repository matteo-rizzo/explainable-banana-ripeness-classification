# Use a pipeline as a high-level helper
from __future__ import annotations

import re

from transformers import pipeline

from src.nlp.ami2020.text_features import separate_html_entities


# Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification


def create_hf_pipeline(model_name: str, device: int | str, batch_size: int = None, top_k=None) -> pipeline:
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)

    pipe = pipeline("text-classification", model=model_name, device=device, batch_size=batch_size, top_k=top_k)
    return pipe


def deep_preprocessing(string: str) -> str:
    # Separate EMOJIS from adjacent words if necessary
    string_clean = separate_html_entities(string)
    # Replace EMOJIS and ENTITIES with codes like ":CODE:"
    # string_clean = replace_with_unicode(string_clean, self.__entity_map)
    # Remove all substrings with < "anything but spaces" >
    string_clean = re.sub("<\S+>", "", string_clean, flags=re.RegexFlag.IGNORECASE).strip()
    # Replace punctuation with space
    # string_clean = re.sub(punctuation, " ", string_clean).strip()
    # Remove double spaces
    string_clean = re.sub(" +", " ", string_clean).strip()

    # Regular expression pattern with negative lookahead (remove all characters that are not A-z, 0-9,
    # and all strings made of ":A-Z:", removing the colons
    # string_clean = re.sub(r"(?!:[A-Z]+:)[^\w\s]|_", "", string_clean)  # removed !? for now
    # string_clean = re.sub(r":", "", string_clean).strip()

    return string_clean
