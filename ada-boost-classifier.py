import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] > self.threshold] = -1
        return predictions


class AdaBoostClassifierScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float("inf")

            # Find best stump
            for feat in range(n_features):
                thresholds = np.unique(X[:, feat])
                for t in thresholds:
                    for polarity in [1, -1]:
                        preds = np.ones(n_samples)
                        if polarity == 1:
                            preds[X[:, feat] < t] = -1
                        else:
                            preds[X[:, feat] > t] = -1

                        error = np.sum(w[y != preds])
                        if error < min_error:
                            min_error = error
                            stump.polarity = polarity
                            stump.threshold = t
                            stump.feature_index = feat

            EPS = 1e-10
            stump.alpha = 0.5 * np.log((1 - min_error) / (min_error + EPS))
            preds = stump.predict(X)

            # Update weights
            w *= np.exp(-stump.alpha * y * preds)
            w /= np.sum(w)

            self.models.append(stump)

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for stump in self.models:
            preds += stump.alpha * stump.predict(X)
        return np.where(preds >= 0, 1, 0)
