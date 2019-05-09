from typing import Dict, List, Union
import logging
import json
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("pairs_reader")
class PairsDatasetReader(DatasetReader):
    def __init__(self,
                 source_language: str = 'en_core_web_sm',
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None) -> None:
        super().__init__()
        self._tokenizer = SpacyWordSplitter(language=source_language)
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                tokens = line.split('\t')

                query_paper = tokens[0].encode("utf-8").decode("unicode_escape")
                candidate_paper = tokens[1].encode("utf-8").decode("unicode_escape")
                relevance = tokens[2]

                instance = self.text_to_instance(query_paper=query_paper, candidate_paper=candidate_paper, relevance=relevance)
                if instance is not None:
                    yield instance

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]
        return tokens

    @overrides
    def text_to_instance(self, query_paper: str, candidate_paper: str, relevance: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        query_tokens = self._tokenizer.split_words(query_paper)
        if self._max_sequence_length is not None:
            query_tokens = self._truncate(query_tokens)
        fields['query_paper'] = TextField(query_tokens, self._token_indexers)

        candidate_tokens = self._tokenizer.split_words(candidate_paper)
        if self._max_sequence_length is not None:
            candidate_tokens = self._truncate(candidate_tokens)
        fields['candidate_paper'] = TextField(candidate_tokens, self._token_indexers)

        fields['label'] = LabelField(relevance)

        fields["metadata"] = MetadataField({"query_paper": query_paper,
                                            "candidate_paper": candidate_paper,
                                            "label": relevance})

        return Instance(fields)
