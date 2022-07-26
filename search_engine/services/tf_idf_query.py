import json
# import nltk
import os
from typing import Union

from nltk import word_tokenize


# nltk.download('stopwords')
# nltk.download('punkt')


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


class TFIDFQueryService:

    def __init__(self):
        with open(os.path.dirname(__file__) + "/../../data-2.json", 'r') as file:
            data = json.load(file)
            self.data = list_to_dict(data, 'paperId')
        self.articles_tf_idf = dict()
        self.unique_words = set()
        self._pre_process()

    def _pre_process(self):
        with open(os.path.dirname(__file__) + "/../../articles_unique_words.json", 'r') as file:
            self.unique_words = set(json.loads(file.read()))

        with open(os.path.dirname(__file__) + "/../../articles_tf_ids.json", 'r') as file:
            self.articles_tf_idf = json.loads(file.read())

    def query(self, query, result_count: int = 3):
        query = word_tokenize(query)
        different_words = []
        result = []
        articles = dict.fromkeys(self.articles_tf_idf.keys(), 0)
        for word in query:
            different_words.append(word.lower())

        for search_word in set(different_words):
            if search_word in self.unique_words:
                for key, data in self.articles_tf_idf.items():
                    articles[key] += data.get(search_word, 0)

        for article, score in articles.items():
            if score > 0:
                result.append((self.data[article], score))

        result.sort(reverse=True, key=lambda x: x[1])
        return self._represent_docs(result[:result_count])

    def _represent_docs(self, results):
        return [
            {
                'paper_title': item[0]['title'],
                'paper_id': item[0]['paperId']
            }
            for item in results
        ]


tf_ids_query_service = TFIDFQueryService()