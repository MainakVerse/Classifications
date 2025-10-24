import numpy as np

class CategoricalNaiveBayesScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.feature_prob_ = {}
        self.class_log_prior_ = None

    def fit(self, X, y):
        """Fit Categorical NB model"""
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        class_count = np.zeros(n_classes)
        self.feature_prob_ = {cls: [] for cls in self.classes_}

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            class_count[idx] = X_c.shape[0]

            for j in range(n_features):
                values, counts = np.unique(X_c[:, j], return_counts=True)
                probs = np.zeros(np.max(X[:, j]) + 1)
                for v, c in zip(values, counts):
                    probs[v] = c
                probs = (probs + self.alpha) / (np.sum(counts) + self.alpha * len(probs))
                self.feature_prob_[cls].append(probs)

        self.class_log_prior_ = np.log(class_count / n_samples)

    def _joint_log_likelihood(self, X):
        jll = []
        for i, cls in enumerate(self.classes_):
            log_likelihood = np.zeros(X.shape[0])
            for j in range(X.shape[1]):
                probs = self.feature_prob_[cls][j]
                feature_vals = X[:, j].astype(int)
                log_likelihood += np.log(probs[feature_vals])
            jll.append(self.class_log_prior_[i] + log_likelihood)
        return np.array(jll).T

    def predict(self, X):
        """Predict class labels"""
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
