import numpy as np

class QuadraticDiscriminantAnalysisScratch:
    def __init__(self):
        self.means_ = {}
        self.covs_ = {}
        self.priors_ = {}
        self.classes_ = None

    def fit(self, X, y):
        """Estimate class means, class-specific covariance matrices, and priors"""
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            X_c = X[y == cls]
            self.means_[cls] = np.mean(X_c, axis=0)
            self.covs_[cls] = np.cov(X_c, rowvar=False)
            self.priors_[cls] = X_c.shape[0] / X.shape[0]

    def _discriminant(self, X, cls):
        mean_vec = self.means_[cls]
        cov = self.covs_[cls]
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        return (
            -0.5 * np.sum(np.dot((X - mean_vec), cov_inv) * (X - mean_vec), axis=1)
            - 0.5 * np.log(cov_det)
            + np.log(self.priors_[cls])
        )

    def predict(self, X):
        """Predict class label"""
        scores = np.array([self._discriminant(X, cls) for cls in self.classes_])
        return self.classes_[np.argmax(scores, axis=0)]
