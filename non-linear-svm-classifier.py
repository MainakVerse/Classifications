import numpy as np

class NonlinearSVMScratch:
    def __init__(self, learning_rate=0.001, C=1.0, n_iters=1000, kernel="rbf", gamma=0.5, degree=3):
        self.lr = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None

    def _kernel(self, x1, x2):
        if self.kernel_type == "linear":
            return np.dot(x1, x2)
        elif self.kernel_type == "poly":
            return (1 + np.dot(x1, x2)) ** self.degree
        elif self.kernel_type == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, X, y):
        """Simplified kernelized SVM training using gradient ascent on dual form"""
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel(X[i], X[j])

        for _ in range(self.n_iters):
            for i in range(n_samples):
                grad = 1 - y[i] * np.sum(self.alpha * y * K[:, i])
                self.alpha[i] += self.lr * grad
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)

        sv = self.alpha > 1e-5
        self.b = np.mean(
            [y[i] - np.sum(self.alpha * y * K[:, i]) for i in range(n_samples) if sv[i]]
        )

    def project(self, X):
        """Compute decision function"""
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for a, y_i, x_i in zip(self.alpha, self.y, self.X):
                if a > 1e-5:
                    s += a * y_i * self._kernel(X[i], x_i)
            y_pred[i] = s
        return y_pred + self.b

    def predict(self, X):
        """Predict class labels"""
        return np.sign(self.project(X))
