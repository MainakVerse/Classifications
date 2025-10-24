import numpy as np

class BayesianNetworkClassifierScratch:
    def __init__(self):
        self.classes_ = None
        self.feature_probs_ = {}
        self.class_priors_ = {}

    def fit(self, X, y):
        """Fit a simple Bayesian network assuming conditional independence"""
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        for cls in self.classes_:
            X_c = X[y == cls]
            self.class_priors_[cls] = len(X_c) / n_samples
            self.feature_probs_[cls] = {}

            for j in range(n_features):
                values, counts = np.unique(X_c[:, j], return_counts=True)
                probs = (counts + 1) / (len(X_c) + len(values))  # Laplace smoothing
                self.feature_probs_[cls][j] = dict(zip(values, probs))

    def _posterior(self, x, cls):
        posterior = np.log(self.class_priors_[cls])
        for j, val in enumerate(x):
            feature_probs = self.feature_probs_[cls].get(j, {})
            posterior += np.log(feature_probs.get(val, 1e-6))  # unseen values
        return posterior

    def predict(self, X):
        """Predict class labels"""
        preds = []
        for x in X:
            posteriors = [self._posterior(x, cls) for cls in self.classes_]
            preds.append(self.classes_[np.argmax(posteriors)])
        return np.array(preds)
