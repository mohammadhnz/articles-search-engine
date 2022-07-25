from typing import List

from elasticsearch_dsl import Q

from search_engine.documents import Article


class ElasticQueryService:
    def query(self, query: str, result_count: int = 3):
        search = Article.search()
        search = search.query(
            Q(
                'bool',
                should=[
                    self._get_match_query("title", query),
                    self._get_match_query("abstract", query),
                    self._get_match_query("authors.name", query)
                ],
                minimum_should_match=1
            )
        )
        results = search.execute()[:result_count]
        return self._represent_docs(results)

    def _get_match_query(self, field, value):
        return {
            "match": {
                field: {
                    "query": value,
                    "operator": "and",
                    "fuzziness": "AUTO"
                }
            }
        }

    def _represent_docs(self, results: List[Article]):
        return [
            {'paper_title': item.title, 'paper_id': item.id}
        for item in results
        ]

elastic_query_service = ElasticQueryService()