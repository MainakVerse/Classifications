import numpy as np

class XGBoostTree:
    def __init__(self, max_depth=3, min_samples_split=2, lam=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lam = lam
        self.is_leaf = False
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.leaf_value = None

    def _calc_gain(self, G_left, H_left, G_right, H_right, G_total, H_total):
        def calc(G, H): return (G ** 2) / (H + self.lam)
        return 0.5 * (calc(G_left, H_left) + calc(G_right, H_right) - calc(G_total, H_total))

    def _best_split(self, X, g, h):
        n_samples, n_features = X.shape
        best_gain, best_feat, best_thresh = -1, None, None
        G_total, H_total = np.sum(g), np.sum(h)

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left = X[:, feat] <= t
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue
                G_left, H_left = np.sum(g[left]), np.sum(h[left])
                G_right, H_right = np.sum(g[right]), np.sum(h[right])
                gain = self._calc_gain(G_left, H_left, G_right, H_right, G_total, H_total)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        return best_feat, best_thresh, best_gain

    def _leaf_value(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.lam)

    def fit(self, X, g, h, depth=0):
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            self.is_leaf = True
            self.leaf_value = self._leaf_value(g, h)
            return

        feat, thresh, gain = self._best_split(X, g, h)
        if feat is None or gain <= 0:
            self.is_leaf = True
            self.leaf_value = self._leaf_value(g, h)
            return

        left = X[:, feat] <= thresh
        right = ~left
        self.split_feature = feat
        self.split_value = thresh
        self.left = XGBoostTree(self.max_depth, self.min_samples_split, self.lam)
        self.right = XGBoostTree(self.max_depth, self.min_samples_split, self.lam)
        self.left.fit(X[left], g[left], h[left], depth + 1)
        self.right.fit(X[right], g[right], h[right], depth + 1)

    def predict(self, X):
        if self.is_leaf:
            return np.full(X.shape[0], self.leaf_value)
        left = X[:, self.split_feature] <= self.split_value
        right = ~left
        preds = np.empty(X.shape[0])
        preds[left] = self.left.predict(X[left])
        preds[right] = self.right.predict(X[right])
        return preds


class XGBoostClassifierScratch:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_split=2, lam=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lam = lam
        self.trees = []
        self.base_score = 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        y = y.astype(float)
        y = np.where(y == 0, 0, 1)
        self.base_score = np.log(np.mean(y) / (1 - np.mean(y) + 1e-10))
        F = np.full(y.shape, self.base_score)

        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            g = p - y                      # Gradient
            h = p * (1 - p)                # Hessian
            tree = XGBoostTree(self.max_depth, self.min_samples_split, self.lam)
            tree.fit(X, g, h)
            update = tree.predict(X)
            F -= self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        F = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            F -= self.learning_rate * tree.predict(X)
        return np.where(self._sigmoid(F) >= 0.5, 1, 0)
