import numpy as np
from copy import deepcopy

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return 1 - np.sum((counts / len(y)) ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        features = np.random.choice(n_features, self.n_features, replace=False)
        best_gain, best_idx, best_thresh = -1, None, None
        parent_gini = self._gini(y)

        for feature_idx in features:
            thresholds = np.unique(X[:, feature_idx])
            for t in thresholds:
                left_idx = X[:, feature_idx] <= t
                right_idx = X[:, feature_idx] > t
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                n_left, n_right = len(y[left_idx]), len(y[right_idx])
                child_gini = (n_left / n_samples) * self._gini(y[left_idx]) + \
                             (n_right / n_samples) * self._gini(y[right_idx])
                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain, best_idx, best_thresh = gain, feature_idx, t
        return best_idx, best_thresh

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))
        if depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return DecisionTreeNode(value=self._most_common_label(y))

        left_idx = X[:, feat_idx] <= threshold
        right_idx = X[:, feat_idx] > threshold
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return DecisionTreeNode(feature_index=feat_idx, threshold=threshold, left=left, right=right)

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestClassifierScratch:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """Train multiple decision trees on random subsets (bagging)"""
        self.trees = []
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """Predict via majority vote"""
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [np.bincount(preds).argmax() for preds in tree_preds]
        return np.array(y_pred)
