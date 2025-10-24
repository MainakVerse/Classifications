import numpy as np
from collections import Counter

class KNearestNeighborsScratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """Predict class labels"""
        preds = [self._predict_single(x) for x in X]
        return np.array(preds)

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_neighbors = self.y_train[k_indices]
        most_common = Counter(k_neighbors).most_common(1)[0][0]
        return most_common
