import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        n_samples, n_features = X.shape
        best_mse = float("inf")

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for t in thresholds:
                left_idx = X[:, feature_idx] <= t
                right_idx = X[:, feature_idx] > t

                if len(residuals[left_idx]) == 0 or len(residuals[right_idx]) == 0:
                    continue

                left_value = np.mean(residuals[left_idx])
                right_value = np.mean(residuals[right_idx])

                y_pred = np.where(left_idx, left_value, right_value)
                mse = np.mean((residuals - y_pred) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    self.feature_index = feature_idx
                    self.threshold = t
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        feature = X[:, self.feature_index]
        return np.where(feature <= self.threshold, self.left_value, self.right_value)


class GradientBoostingClassifierScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.F0 = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """Train gradient boosting with logistic loss"""
        y = np.where(y == 0, -1, 1)
        self.F0 = np.log((1 + np.mean(y)) / (1 - np.mean(y))) if np.mean(y) != 0 else 0
        F = np.full(y.shape, self.F0)

        for _ in range(self.n_estimators):
            p = self._sigmoid(2 * F)
            residuals = y * (1 - p)
            stump = DecisionStump()
            stump.fit(X, residuals)
            self.models.append(stump)
            F += self.learning_rate * stump.predict(X)

    def predict(self, X):
        F = np.full((X.shape[0],), self.F0)
        for stump in self.models:
            F += self.learning_rate * stump.predict(X)
        return np.where(self._sigmoid(2 * F) >= 0.5, 1, 0)
