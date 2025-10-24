import numpy as np
from collections import Counter

class RadiusNeighborsClassifierScratch:
    def __init__(self, radius=1.0, outlier_label=None):
        self.radius = radius
        self.outlier_label = outlier_label
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
        neighbors_idx = [i for i, d in enumerate(distances) if d <= self.radius]

        if len(neighbors_idx) == 0:
            return self.outlier_label if self.outlier_label is not None else self._majority_class()

        neighbor_labels = self.y_train[neighbors_idx]
        most_common = Counter(neighbor_labels).most_common(1)[0][0]
        return most_common

    def _majority_class(self):
        """Return the majority class of the training data (for outliers)"""
        return Counter(self.y_train).most_common(1)[0][0]
