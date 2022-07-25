import json
import os.path
import time

from sentence_transformers import CrossEncoder


class TransformerQueryService:
    def __init__(self):
        with open(os.path.dirname(__file__) + '/../../transformer_engine_data.json', 'r') as file:
            self.data = json.load(file)
        self.model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

    def query(self, query, result_count: int = 3):
        scores = []
        for item in self.data:
            try:
                document = item['abstract']
                scores.append(self._get_query_score_in_doc(document, query))
            except:
                scores.append(0)
        results = [{'paper': paper, 'score': score} for paper, score in
                   zip(self.data, scores)]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return self._represent_docs(results[:result_count])

    def _get_query_score_in_doc(self, passages, query):
        model_inputs = [[query, passage] for passage in passages]
        scores = self.model.predict(model_inputs)
        return max(scores)

    def _represent_docs(self, results):
        return [
            {
                'paper_title': item['paper']['title'],
                'paper_id': item['paper']['paperId']
            }
            for item in results
        ]


transformer_query_service = TransformerQueryService()
