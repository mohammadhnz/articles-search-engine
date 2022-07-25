from django.shortcuts import render
from rest_framework.views import APIView
from .services import elastic_query_service, boolean_query_service
from .services import cluster_service, classification_service


class QueryView(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request, format=None):
        query = request.query_params.get('query')
        if not query:
            return render(request, 'search.html', {'results': {}})
        results = {
            'Elastic Query': elastic_query_service.query(query),
            'Boolean Query': boolean_query_service.query(query),
            'Transformer Query': transformer_query_service.query(query),
        }
        classification_result = classification_service.predict_text_class(query)
        clustering_result = cluster_service.predict_text_cluster([query])
        print(classification_result)
        print(clustering_result)
        return render(
            request, 'search.html', {
                'results': results,
                'query': query,
                'classification_result': classification_result,
                'clustering_result': clustering_result
            }
        )
