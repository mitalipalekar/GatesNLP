from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import json


@Predictor.register('relevance_predictor')
class RelevancePredictor(Predictor):
    '''
    Basic predictor class for use with the SentimentPlaintextReader.

    Usage:
      allennlp predict \
        <model_path> <input_path> \
        --include-package <package_name> \
        --predictor sentiment_predictor \
        --overrides "{dataset_reader: {type: 'sentiment_plaintext_reader'}}"
    '''

    def predict(self, query_paper: str, candidate_paper: str) -> JsonDict:
        return self.predict_json({'query_paper': query_paper, 'candidate_paper': candidate_paper})

    @overrides
    def load_line(self, line: str) -> JsonDict:
        # Since we don't have any input fields besides the sentence itself,
        # it doesn't really make sense to pack things in json -- just have each
        # input line be a sentence we want to classify.
        json_dict = json.loads(line)
        return {"query_paper": json_dict['query_paper'], "candidate_paper": json_dict['candidate_paper']}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(json_dict['query_paper'], json_dict['candidate_paper'])
        return instance
