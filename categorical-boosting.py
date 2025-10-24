import numpy as np

class CatBoostTree:
    def __init__(self, max_depth=3, learning_rate=0.1):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X, residuals, depth=0):
        if depth >= self.max_depth or len(residuals) <= 1:
            self.value = np.mean(residuals)
            return

        best_feat, best_thresh, best_loss = None, None, float("inf")
        n_samples, n_features = X.shape

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left = X[:, feat] <= t
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue
                left_mean = np.mean(residuals[left])
                right_mean = np.mean(residuals[right])
                preds = np.where(left, left_mean, right_mean)
                loss = np.mean((residuals - preds) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    best_feat = feat
                    best_thresh = t

        if best_feat is None:
            self.value = np.mean(residuals)
            return

        self.feature_index = best_feat
        self.threshold = best_thresh
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        self.left = CatBoostTree(self.max_depth, self.learning_rate)
        self.right = CatBoostTree(self.max_depth, self.learning_rate)
        self.left.fit(X[left_idx], residuals[left_idx], depth + 1)
        self.right.fit(X[right_idx], residuals[right_idx], depth + 1)

    def predict(self, X):
        if self.value is not None:
            return np.full(X.shape[0], self.value)
        preds = np.zeros(X.shape[0])
        left_idx = X[:, self.feature_index] <= self.threshold
        right_idx = X[:, self.feature_index] > self.threshold
        preds[left_idx] = self.left.predict(X[left_idx])
        preds[right_idx] = self.right.predict(X[right_idx])
        return preds


class CatBoostClassifierScratch:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_score = 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _ordered_target_encoding(self, X, y):
        """Encode categorical features using cumulative mean (ordered boosting trick)"""
        X_encoded = np.copy(X).astype(float)
        for col in range(X.shape[1]):
            if np.issubdtype(X[:, col].dtype, np.integer):
                running_sum, running_count = 0, 0
                encoded = []
                for val, target in zip(X[:, col], y):
                    mean_enc = running_sum / running_count if running_count > 0 else np.mean(y)
                    encoded.append(mean_enc)
                    running_sum += target
                    running_count += 1
                X_encoded[:, col] = encoded
        return X_encoded

    def fit(self, X, y):
        y = np.where(y == 0, 0, 1)
        X = self._ordered_target_encoding(X, y)
        self.base_score = np.log(np.mean(y) / (1 - np.mean(y) + 1e-10))
        F = np.full(y.shape, self.base_score)

        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            residuals = y - p
            tree = CatBoostTree(max_depth=self.max_depth, learning_rate=self.learning_rate)
            tree.fit(X, residuals)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)

    def predict(self, X):
        X = np.copy(X).astype(float)
        F = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return np.where(self._sigmoid(F) >= 0.5, 1, 0)
