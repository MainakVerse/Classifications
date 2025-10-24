import numpy as np

class RidgeClassifierScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Fit ridge regression model for classification"""
        # Convert labels to {-1, 1}
        y = np.where(y == 0, -1, 1)
        n_samples, n_features = X.shape

        # Closed-form solution: w = (XᵀX + αI)⁻¹Xᵀy
        I = np.eye(n_features)
        self.weights = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y
        self.bias = np.mean(y - np.dot(X, self.weights))

    def predict(self, X):
        """Predict binary class labels"""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
