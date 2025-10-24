import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Build the decision tree recursively"""
        self.root = self._grow_tree(X, y)

    def _gini(self, y):
        """Compute Gini impurity"""
        classes, counts = np.unique(y, return_counts=True)
        prob_sq_sum = np.sum((counts / len(y)) ** 2)
        return 1 - prob_sq_sum

    def _best_split(self, X, y):
        """Find the best split for a node"""
        n_samples, n_features = X.shape
        best_gain, best_idx, best_thresh = -1, None, None
        parent_gini = self._gini(y)

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for t in thresholds:
                left_idx = X[:, feature_idx] <= t
                right_idx = X[:, feature_idx] > t
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                # Weighted Gini
                n_left, n_right = len(y[left_idx]), len(y[right_idx])
                child_gini = (n_left / n_samples) * self._gini(y[left_idx]) + \
                             (n_right / n_samples) * self._gini(y[right_idx])
                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain, best_idx, best_thresh = gain, feature_idx, t

        return best_idx, best_thresh

    def _grow_tree(self, X, y, depth=0):
        """Recursive tree growth"""
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        left_idx = X[:, feat_idx] <= threshold
        right_idx = X[:, feat_idx] > threshold
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return DecisionTreeNode(feature_index=feat_idx, threshold=threshold, left=left, right=right)

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        """Predict class labels for samples"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse the decision tree"""
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
