import numpy as np

class NuSupportVectorClassifierScratch:
    def __init__(self, nu=0.5, kernel="linear", gamma=0.5, degree=3, n_iters=1000, lr=0.001):
        self.nu = nu
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.n_iters = n_iters
        self.lr = lr
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
        """Train simplified Nu-SVC"""
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)

        # Compute kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel(X[i], X[j])

        # Initialize target sum constraint for ν
        target_sum = self.nu * n_samples

        for _ in range(self.n_iters):
            for i in range(n_samples):
                grad = 1 - y[i] * np.sum(self.alpha * y * K[:, i])
                self.alpha[i] += self.lr * grad

                # Enforce ν constraint and positivity
                self.alpha[i] = np.clip(self.alpha[i], 0, 1.0 / n_samples)
                if np.sum(self.alpha) > target_sum:
                    self.alpha *= target_sum / np.sum(self.alpha)

        # Compute bias (b)
        sv = self.alpha > 1e-5
        self.b = np.mean(
            [y[i] - np.sum(self.alpha * y * K[:, i]) for i in range(n_samples) if sv[i]]
        )

    def project(self, X):
        """Compute decision function"""
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for a, y_i, x_i in zip(self.alpha, self.y, self.X):
                if a > 1e-5:
                    s += a * y_i * self._kernel(X[i], x_i)
            result[i] = s
        return result + self.b

    def predict(self, X):
        """Predict class labels"""
        return np.sign(self.project(X))
