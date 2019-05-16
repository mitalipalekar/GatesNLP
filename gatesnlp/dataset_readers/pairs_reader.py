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
                line_json = json.loads(line)
                if not line_json:
                    continue

                query_paper = line_json["query_paper"]
                candidate_paper = line_json["candidate_paper"]
                relevance = line_json["relevance"]

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

        if relevance is not None:
            fields['label'] = LabelField(relevance)

        return Instance(fields)
