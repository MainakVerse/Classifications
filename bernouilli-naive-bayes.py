import numpy as np

class BernoulliNaiveBayesScratch:
    def __init__(self, alpha=1.0, binarize_threshold=0.5):
        self.alpha = alpha
        self.binarize_threshold = binarize_threshold
        self.classes_ = None
        self.feature_log_prob_ = None
        self.feature_log_neg_prob_ = None
        self.class_log_prior_ = None

    def _binarize(self, X):
        return np.where(X > self.binarize_threshold, 1, 0)

    def fit(self, X, y):
        """Fit Bernoulli NB model"""
        X = self._binarize(X)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        class_count = np.zeros(n_classes)
        feature_prob = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            class_count[idx] = X_c.shape[0]
            feature_prob[idx, :] = (np.sum(X_c, axis=0) + self.alpha) / (X_c.shape[0] + 2 * self.alpha)

        self.class_log_prior_ = np.log(class_count / n_samples)
        self.feature_log_prob_ = np.log(feature_prob)
        self.feature_log_neg_prob_ = np.log(1 - feature_prob)

    def _joint_log_likelihood(self, X):
        """Compute joint log likelihood"""
        return (
            X @ self.feature_log_prob_.T
            + (1 - X) @ self.feature_log_neg_prob_.T
            + self.class_log_prior_
        )

    def predict(self, X):
        """Predict class labels"""
        X = self._binarize(X)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
