import numpy as np

class LinearDiscriminantAnalysisScratch:
    def __init__(self):
        self.means_ = {}
        self.priors_ = {}
        self.cov_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Estimate class means, shared covariance, and priors"""
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.cov_ = np.zeros((n_features, n_features))

        # Compute mean and prior for each class
        for cls in self.classes_:
            X_c = X[y == cls]
            self.means_[cls] = np.mean(X_c, axis=0)
            self.priors_[cls] = X_c.shape[0] / X.shape[0]
            self.cov_ += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)

        # Shared covariance (pooled)
        self.cov_ /= (X.shape[0] - len(self.classes_))

        # Precompute inverse for efficiency
        self.cov_inv_ = np.linalg.inv(self.cov_)

    def _discriminant(self, X, cls):
        """Compute discriminant function for class"""
        mean_vec = self.means_[cls]
        return (
            np.dot(X, np.dot(self.cov_inv_, mean_vec))
            - 0.5 * np.dot(mean_vec.T, np.dot(self.cov_inv_, mean_vec))
            + np.log(self.priors_[cls])
        )

    def predict(self, X):
        """Predict class label"""
        scores = [self._discriminant(X, cls) for cls in self.classes_]
        return self.classes_[np.argmax(scores, axis=0)]
