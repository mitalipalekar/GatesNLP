from typing import Dict, List, Union
import logging
import json
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("gatesnlp_dataset_reader")
class GatesNLPReader(DatasetReader):
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
                items = json.loads(line)
                paper_id = items["id"]
                title = items["title"]
                paperAbstract = items["paperAbstract"]
                outCitations = items["outCitations"]
                
                instance = self.text_to_instance(paper_id=paper_id, titles=title, paperAbstract=paperAbstract, outCitations=outCitations)
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
    def text_to_instance(self, paper_id: str, titles: str, paperAbstract: str, outCitations: List[str]) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        title_tokens = self._tokenizer.split_words(titles)
        if self._max_sequence_length is not None:
            tokens = self._truncate(title_tokens)
        fields['title'] = TextField(tokens, self._token_indexers)

        abtract_tokens = self._tokenizer.split_words(paperAbstract)
        if self._max_sequence_length is not None:
            tokens = self._truncate(abtract_tokens)
        fields['abstract'] = TextField(tokens, self._token_indexers)


        fields["metadata"] = MetadataField({"id": paper_id, 
                                            "title": titles,
                                            "abstract": paperAbstract,
                                            "citation": outCitations})

        # print fields and token
        print(fields['metadata']['id'])
        print(fields['title'])
        print(fields['abstract'])
        print(fields['metadata']['citation'])

        return Instance(fields)
