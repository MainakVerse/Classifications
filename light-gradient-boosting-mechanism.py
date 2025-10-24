import numpy as np
import heapq

class LightGBMNode:
    def __init__(self, grad_sum=0, hess_sum=0, prediction=0, left=None, right=None, feature=None, threshold=None, depth=0):
        self.grad_sum = grad_sum
        self.hess_sum = hess_sum
        self.prediction = prediction
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.depth = depth


class LightGBMTree:
    def __init__(self, max_depth=3, min_data_in_leaf=10, lambda_l2=1):
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.root = None

    def _gain(self, G_left, H_left, G_right, H_right, G_total, H_total):
        def score(G, H): return (G ** 2) / (H + self.lambda_l2)
        return 0.5 * (score(G_left, H_left) + score(G_right, H_right) - score(G_total, H_total))

    def _leaf_value(self, G, H):
        return -G / (H + self.lambda_l2)

    def _best_split(self, X, g, h):
        n_samples, n_features = X.shape
        G_total, H_total = np.sum(g), np.sum(h)
        best_gain, best_feat, best_thresh = -1, None, None

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left = X[:, feat] <= t
                right = ~left
                if np.sum(left) < self.min_data_in_leaf or np.sum(right) < self.min_data_in_leaf:
                    continue

                G_left, H_left = np.sum(g[left]), np.sum(h[left])
                G_right, H_right = np.sum(g[right]), np.sum(h[right])
                gain = self._gain(G_left, H_left, G_right, H_right, G_total, H_total)

                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        return best_feat, best_thresh, best_gain

    def fit(self, X, g, h):
        G, H = np.sum(g), np.sum(h)
        self.root = LightGBMNode(grad_sum=G, hess_sum=H)
        candidates = [(0, self.root, X, g, h)]
        heapq.heapify(candidates)

        while candidates:
            _, node, X_node, g_node, h_node = heapq.heappop(candidates)

            if node.depth >= self.max_depth or len(g_node) < 2 * self.min_data_in_leaf:
                node.prediction = self._leaf_value(np.sum(g_node), np.sum(h_node))
                continue

            feat, thresh, gain = self._best_split(X_node, g_node, h_node)
            if feat is None or gain <= 0:
                node.prediction = self._leaf_value(np.sum(g_node), np.sum(h_node))
                continue

            left = X_node[:, feat] <= thresh
            right = ~left
            node.feature = feat
            node.threshold = thresh

            node.left = LightGBMNode(depth=node.depth + 1)
            node.right = LightGBMNode(depth=node.depth + 1)

            heapq.heappush(candidates, (-gain, node.left, X_node[left], g_node[left], h_node[left]))
            heapq.heappush(candidates, (-gain, node.right, X_node[right], g_node[right], h_node[right]))

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.left is None and node.right is None:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)


class LightGBMClassifierScratch:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_data_in_leaf=10, lambda_l2=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.trees = []
        self.base_score = 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        y = np.where(y == 0, 0, 1)
        self.base_score = np.log(np.mean(y) / (1 - np.mean(y) + 1e-10))
        F = np.full(y.shape, self.base_score)

        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            g = p - y
            h = p * (1 - p)
            tree = LightGBMTree(self.max_depth, self.min_data_in_leaf, self.lambda_l2)
            tree.fit(X, g, h)
            self.trees.append(tree)
            F -= self.learning_rate * tree.predict(X)

    def predict(self, X):
        F = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            F -= self.learning_rate * tree.predict(X)
        return np.where(self._sigmoid(F) >= 0.5, 1, 0)
