import os
import pickle


class ClusterService:
    def __init__(self):
        self.model, self.vectorizer = self.load_model()

    def load_model(self):
        model = pickle.load(open(os.path.dirname(__file__) + "/../../model.pkl", "rb"))
        vectorizer = pickle.load(open(os.path.dirname(__file__) + "/../../vectorizer.pickle", "rb"))
        return model, vectorizer

    def predict_text_cluster(self, text):
        vectorized = self.vectorizer.transform(text)
        return self._represent_result(self.model.predict(vectorized))

    def _represent_result(self, result):
        order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names()
        i = result.tolist()[0]
        fitures = []
        for ind in order_centroids[i, :10]:
            fitures.append(terms[ind])
        return {"cluster_id": i, 'priority_score': 0.647, "keywords": fitures}


cluster_service = ClusterService()
