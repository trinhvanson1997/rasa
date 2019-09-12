from __future__ import absolute_import, division, print_function

import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

import numpy as np
from rasa_nlu import utils
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message, TrainingData
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class TfidfFeaturizer(Featurizer):
    provides = ["text_features"]

    requires = []

    defaults = {
        # TfidfVectorizer's params
        "strip_accents": None,
        "lowercase": True,
        "analyzer": "word",
        "stop_words": None,
        "token_pattern": u"\w+",
        "min_ngram": 1,
        "max_ngram": 2,
        "min_df": 3,
        "max_df": 0.8,
        "max_features": None,
        "vocabulary": None,
        "binary": False,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True,
        # out-of-words
        "oov_token": None,
        "oov_words": [],
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super(TfidfFeaturizer, self).__init__(component_config)

        self.strip_accents = self.component_config["strip_accents"]
        self.lowercase = self.component_config["lowercase"]
        self.analyzer = self.component_config["analyzer"]
        self.stop_words = self.component_config["stop_words"]
        self.token_pattern = self.component_config["token_pattern"]
        self.ngram_range = (
            self.component_config["min_ngram"],
            self.component_config["max_ngram"],
        )
        self.min_df = self.component_config["min_df"]
        self.max_df = self.component_config["max_df"]
        self.max_features = self.component_config["max_features"]
        self.vocabulary = self.component_config["vocabulary"]
        self.binary = self.component_config["binary"]
        self.norm = self.component_config["norm"]
        self.use_idf = self.component_config["use_idf"]
        self.smooth_idf = self.component_config["smooth_idf"]
        self.sublinear_tf = self.component_config["sublinear_tf"]
        self.oov_token = self.component_config["oov_token"]
        self.oov_words = self.component_config["oov_words"]

        if self.oov_words and not self.oov_token:
            logger.error(
                "The list OOV_words={} was given, but "
                "OOV_token was not. OOV words are ignored."
                "".format(self.oov_words)
            )

            self.oov_words = []

        if self.lowercase and self.oov_token:
            # convert to lowercase
            self.oov_token = self.oov_token.lower()
            self.oov_words = [w.lower() for w in self.oov_words]

        # check analyzer
        if self.analyzer != "word":
            if self.oov_token is not None:
                logger.warning(
                    "Analyzer is set to character, "
                    "provided OOV word token will be ignored."
                )
            if self.stop_words is not None:
                logger.warning(
                    "Analyzer is set to character, "
                    "provided stop words will be ignored."
                )
            if self.ngram_range[1] == 1:  # max_ngram
                logger.warning(
                    "Analyzer is set to character, "
                    "but max n-gram is set to 1. "
                    "It means that the vocabulary will "
                    "contain single letters only."
                )

        self.tfidf = None

    @classmethod
    def required_packages(cls):  # type: () -> List[Text]
        return ["sklearn"]

    def tokenizer(self, text: Text) -> List[Text]:
        # split text to tokens
        tokens = re.compile(self.token_pattern).findall(text.strip())

        if self.oov_token:
            if hasattr(self.tfidf, "vocabulary_"):
                # TfidfVectorizer is trained, process for prediction
                if self.oov_token in self.tfidf.vocabulary_:
                    tokens = [
                        t if t in self.tfidf.vocabulary_.keys() else self.oov_token
                        for t in tokens
                    ]
            elif self.oov_words:
                # TfidfVectorizer is not trained, process for train
                tokens = [self.oov_token if t in self.oov_words else t for t in tokens]

        return tokens

    # noinspection PyPep8Naming
    def check_oov_present(self, examples: List[Text]) -> None:
        if self.oov_token and not self.oov_words:
            for t in examples:
                if self.oov_token in t or (
                        self.lowercase and self.oov_token in t.lower()
                ):
                    return
            logger.warning(
                "OOV_token='{}' was given, but it is not present "
                "in the training data. All unseen words "
                "will be ignored during prediction."
                "".format(self.oov_token)
            )

    @staticmethod
    def get_message_text(message: Message) -> Text:
        if message.get("spacy_doc"):  # if lemmatize is possible
            return " ".join([t.lemma_ for t in message.get("spacy_doc")])
        elif message.get("tokens"):  # if directly tokens is provided
            return " ".join([t.text for t in message.get("tokens")])
        else:
            return message.text

    def train(
            self, training_data, cfg, **kwargs
    ):  # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        spacy_nlp = kwargs.get("spacy_nlp")
        if spacy_nlp is not None:
            # create spacy lemma_ for OOV_words
            self.oov_words = [t.lemma_ for w in self.oov_words for t in spacy_nlp(w)]

        # define tfidfvectorizer
        self.tfidf = TfidfVectorizer(
            strip_accents=self.strip_accents,
            lowercase=self.lowercase,
            tokenizer=self.tokenizer,
            analyzer=self.analyzer,
            stop_words=self.stop_words,
            token_pattern=self.token_pattern,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            vocabulary=self.vocabulary,
            binary=self.binary,
            dtype=np.float32,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )

        lem_exs = [
            self.get_message_text(example) for example in training_data.intent_examples
        ]

        self.check_oov_present(lem_exs)

        try:
            # noinspection PyPep8Naming
            X = self.tfidf.fit_transform(lem_exs).toarray()
        except ValueError:
            self.tfidf = None
            return

        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            example.set(
                "text_features",
                self._combine_with_existing_text_features(example, X[i]),
            )

    def process(self, message, **kwargs):  # type: (Message, **Any) -> None
        if self.tfidf is None:
            logger.error(
                "There is no trained TfidfFeaturizer: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
        else:
            message_text = self.get_message_text(message)

            bag = self.tfidf.transform([message_text]).toarray().squeeze()
            message.set(
                "text_features", self._combine_with_existing_text_features(message, bag)
            )

    def persist(self, file_name, model_dir):  # type: (Text, Text) -> Optional[Dict[Text, Any]]
        file_name = self.__class__.__name__ + ".pkl"  # TfidfFeaturizer.pkl

        utils.pycloud_pickle(os.path.join(model_dir, file_name), self)

        return {"file": file_name}

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any] = None,
            model_dir: Text = None,
            model_metadata: Metadata = None,
            cached_component: Optional["TfidfFeaturizer"] = None,
            **kwargs: Any
    ) -> "TfidfFeaturizer":
        if cached_component:
            return cached_component

        else:
            return utils.pycloud_unpickle(
                os.path.join(model_dir, cls.__name__ + ".pkl")
            )
