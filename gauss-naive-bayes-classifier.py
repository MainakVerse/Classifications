import numpy as np

class GaussianNaiveBayesScratch:
    def __init__(self):
        self.classes_ = None
        self.means_ = {}
        self.vars_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        """Compute class-wise mean, variance, and prior probabilities"""
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            X_c = X[y == cls]
            self.means_[cls] = np.mean(X_c, axis=0)
            self.vars_[cls] = np.var(X_c, axis=0)
            self.priors_[cls] = X_c.shape[0] / X.shape[0]

    def _gaussian_pdf(self, class_idx, x):
        mean = self.means_[class_idx]
        var = self.vars_[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var + 1e-9))
        denominator = np.sqrt(2 * np.pi * var + 1e-9)
        return numerator / denominator

    def _posterior(self, x, cls):
        prior = np.log(self.priors_[cls])
        conditional = np.sum(np.log(self._gaussian_pdf(cls, x)))
        return prior + conditional

    def predict(self, X):
        """Predict class labels"""
        preds = [self._predict_single(x) for x in X]
        return np.array(preds)

    def _predict_single(self, x):
        posteriors = [self._posterior(x, cls) for cls in self.classes_]
        return self.classes_[np.argmax(posteriors)]
