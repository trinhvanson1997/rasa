from __future__ import absolute_import, division, print_function

import json
import logging
import re
from typing import Any, Dict, List, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.training_data import Message, TrainingData
from underthesea import word_tokenize

logger = logging.getLogger(__name__)


class Token(object):
    def __init__(self, text, offset, data=None):
        self.offset = offset
        self.text = text
        self.end = offset + len(text)
        self.data = data if data else {}

    def set(self, prop, info):
        self.data[prop] = info

    def get(self, prop, default=None):
        return self.data.get(prop, default)

    def __repr__(self):
        return "{} (match: {})".format(self.text, (self.offset, self.end))


class ViTokenizer(Component):  # neednt persist
    provides = ["tokens"]

    language_list = ["vi", "vi_spacy_model", "vi_fasttext"]

    defaults = {
        "correct_mapping": "word_mapping.json",
        "lowercase": True,
        "replace_tokens": False,
        "use_punctuation": False,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super(ViTokenizer, self).__init__(component_config)

        # convert tokens to lowercase
        self.lowercase = self.component_config["lowercase"]

        # whether replace numbers by __NUMBER__ and punctuations by __PUNC__
        self.replace_tokens = self.component_config["replace_tokens"]

        self.use_punctuation = self.component_config["use_punctuation"]

        try:
            self.correct_mapping = json.load(
                open(self.component_config["correct_mapping"], "r")
            )

        except Exception as e:
            self.correct_mapping = {}

    @classmethod
    def required_packages(cls):  # type: () -> List[Text]
        return ["underthesea"]

    def tokenize(self, text: Text) -> List[Token]:
        # preprocess
        # - replace any numbers by __NUMBER__ token
        text = (
            re.sub(u"\$?#?\d+(?:\.\d+)?%?", " __NUMBER__ ", text.strip())
            if self.replace_tokens
            else text.strip()
        )
        # - correct words (by mapping)
        text = " ".join(
            [
                self.correct_mapping.get(word.strip().lower(), word.strip())
                for word in text.split(" ")
            ]
        )
        # - tokenize
        text = word_tokenize(text, format="text")
        # - replace any punctuation by __PUNC__ token (skip for "_")
        if self.replace_tokens or not self.use_punctuation:
            text = re.sub(u"\.\.\.|[][.,;\"'?():`~\-!@]", " __PUNC__ ", text)

            if not self.use_punctuation:
                text = re.sub("__PUNC__", "", text)

        # - replace duplicate space
        text = re.sub(" {2,}", " ", text.strip())

        # build tokens list
        tokens = []
        offset = 0
        for word in text.split(" "):
            tokens.append(Token(word.lower() if self.lowercase else word, offset))

            offset += len(word) + 1  # space

        logger.debug("tokens: {}".format(tokens))

        return tokens

    def train(
        self, training_data, cfg, **kwargs
    ):  # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):  # type: (Message, **Any) -> None
        message.set("tokens", self.tokenize(message.text))
