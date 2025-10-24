import numpy as np

class ComplementNaiveBayesScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.feature_log_prob_ = None
        self.class_log_prior_ = None

    def fit(self, X, y):
        """Fit Complement NB model"""
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            class_count[idx] = X_c.shape[0]
            feature_count[idx, :] = np.sum(X_c, axis=0)

        # Complement counts
        total_fc = np.sum(feature_count, axis=0)
        comp_fc = total_fc - feature_count
        smoothed_fc = comp_fc + self.alpha
        smoothed_cc = np.sum(smoothed_fc, axis=1).reshape(-1, 1)

        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)
        self.class_log_prior_ = np.log(class_count / n_samples)

    def _joint_log_likelihood(self, X):
        """Compute joint log likelihood"""
        return -(X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X):
        """Predict class labels"""
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmin(jll, axis=1)]
