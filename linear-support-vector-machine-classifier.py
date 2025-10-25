import numpy as np

class LinearSVMScratch:
    def __init__(self, learning_rate=0.001, C=1.0, n_iters=1000):
        self.lr = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Train a linear SVM classifier"""
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            margins = y_ * (np.dot(X, self.weights) + self.bias)
            mask = margins < 1  # misclassified or within margin
            dw = self.weights - self.C * np.dot(X.T, y_ * mask) / n_samples
            db = -self.C * np.sum(y_ * mask) / n_samples
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """Predict class labels"""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)
