import numpy as np
from collections import Counter
from copy import deepcopy

class BaggingClassifierScratch:
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0):
        """
        base_estimator: any model class with fit() and predict()
        n_estimators: number of models in the ensemble
        max_samples: fraction of samples used for each bootstrap dataset
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []

    def fit(self, X, y):
        """Train multiple estimators on bootstrap samples"""
        n_samples = X.shape[0]
        sample_size = int(self.max_samples * n_samples)
        self.models = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, sample_size, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            model = deepcopy(self.base_estimator)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        """Predict via majority voting"""
        all_preds = np.array([model.predict(X) for model in self.models])
        all_preds = np.swapaxes(all_preds, 0, 1)
        y_pred = [Counter(row).most_common(1)[0][0] for row in all_preds]
        return np.array(y_pred)
