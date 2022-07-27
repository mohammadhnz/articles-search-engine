import json
import os
import re
from typing import Union
import fasttext
import numpy as np
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity


def list_to_dict(data_list: list, keys: Union[str, tuple], distinct=True) -> dict:
    data_dict = {}

    for data in data_list:
        if isinstance(keys, str):
            _key = data.get(keys)
        else:
            _key = tuple(data.get(k) for k in keys)

        _val = data

        if distinct:
            data_dict[_key] = _val
        else:
            if _key not in data_dict:
                data_dict[_key] = [_val]
            else:
                data_dict[_key].append(_val)

    return data_dict


class WORDEMBEDDINGQueryService:

    def __init__(self):
        with open(os.path.dirname(__file__) + "/../../data-2.json", 'r') as file:
            data = json.load(file)
            self.data = list_to_dict(data, 'paperId')
        self.model = None
        self.average_vector = None
        self.articles = []
        self.articles_ids = {}
        self.articles_index_map = {}
        self._pre_process()

    def _remove_special_characters(self, text):
        regex = re.compile('[^a-zA-Z0-9\s]')
        text_returned = re.sub(regex, '', text)
        return text_returned

    def _prepare_text(self, text):
        if not text:
            return []
        text = self._remove_special_characters(text)
        text = re.sub(re.compile('\d'), '', text)
        sentences = sent_tokenize(text)
        return [sentence.lower() for sentence in sentences]

    def _pre_process(self):
        i = 0
        for key, item in self.data.items():
            sentences = self._prepare_text(item['abstract'])
            self.articles += sentences
            for sentence in sentences:
                self.articles_ids[sentence] = item['paperId']
                self.articles_index_map[sentence] = i
                i += 1

        self.model = fasttext.load_model(os.path.dirname(__file__) + '/../../articles_model.bin')
        self.average_vector = np.load(os.path.dirname(__file__) + '/../../articles_average_vector.npy')

    def calculate_average_vector(self, article):
        all_tokens = article.split(' ')
        tokens_vectors = []
        for word in all_tokens:
            tokens_vectors.append(self.model.get_word_vector(word))

        return np.mean(np.array(tokens_vectors), axis=0)

    def find_k_most_relevant(self, query_similarity_vector, k):
        results = []
        for i in range(k):
            max_value = max(query_similarity_vector)
            max_index = query_similarity_vector.index(max_value)
            results.append(self.data[self.articles_ids[self.articles[max_index]]])
            query_similarity_vector[max_index] = -1
        return results

    def prepare_query(self, query):
        if not query:
            return ''

        query = self._remove_special_characters(query)
        return re.sub(re.compile('\d'), '', query)

    def expand_query(self, q, data, count):
        near = data[:10]
        far = data[10:]
        vectors = []
        for a in near:
            for b in self._prepare_text(a['abstract']):
                vectors.append(self.average_vector[self.articles_index_map[b]])
        q = q + np.mean(vectors, axis=0)
        vectors = []
        for a in far:
            for b in self._prepare_text(a['abstract']):
                vectors.append(self.average_vector[self.articles_index_map[b]])
        q = q - np.mean(vectors, axis=0)
        query_similarity_vector = cosine_similarity(
            [q],
            self.average_vector
        )
        query_similarity_vector = list(query_similarity_vector[0])
        results = []
        for i in range(count):
            max_value = max(query_similarity_vector)
            max_index = query_similarity_vector.index(max_value)
            results.append(self.data[self.articles_ids[self.articles[max_index]]])
            query_similarity_vector[max_index] = -1
        return results

    def query(self, query, result_count: int = 3, with_expand=False):
        query = self.prepare_query(query)
        query_vector = self.calculate_average_vector(query)
        query_similarity_vector = cosine_similarity(
            [query_vector],
            self.average_vector
        )
        query_similarity_vector = list(query_similarity_vector[0])
        data1 = self.find_k_most_relevant(query_similarity_vector, result_count)
        data2 = None
        if with_expand:
            data2 = self.find_k_most_relevant(query_similarity_vector, len(query_similarity_vector))
            data2 = self.expand_query(query_vector, data2, result_count)
        return self._represent_docs(data1), self._represent_docs(data2)

    def _represent_docs(self, results):
        return [
            {
                'paper_title': item['title'],
                'paper_id': item['paperId']
            }
            for item in results
        ]


word_embedding_query_service = WORDEMBEDDINGQueryService()