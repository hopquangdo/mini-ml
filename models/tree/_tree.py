import numpy as np


class Node:
    def __init__(self, depth, value=None, feature_index=None, threshold=None, left=None, right=None):
        self.depth = depth
        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right


class TreeBuilder:
    def __init__(self, impurity_func, leaf_func, max_depth, min_samples_split,
                 use_all_thresholds=False):
        self.impurity_func = impurity_func
        self.leaf_func = leaf_func
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.use_all_thresholds = use_all_thresholds

    def build(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split):
            return Node(depth, value=self.leaf_func(y))

        best_feat, best_thresh, best_score = None, None, float('inf')
        for feature_index in range(n_features):
            column = X[:, feature_index]
            values = np.sort(np.unique(column))
            thresholds = values if self.use_all_thresholds else (values[:-1] + values[1:]) / 2
            for threshold in thresholds:
                left_mask = column <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                score = self.impurity_func(y[left_mask], y[right_mask], y)
                if score < best_score:
                    best_score = score
                    best_feat = feature_index
                    best_thresh = threshold

        if best_feat is None:
            return Node(depth, value=self.leaf_func(y))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left = self.build(X[left_mask], y[left_mask], depth + 1)
        right = self.build(X[right_mask], y[right_mask], depth + 1)

        return Node(depth=depth, feature_index=best_feat, threshold=best_thresh, left=left, right=right)
