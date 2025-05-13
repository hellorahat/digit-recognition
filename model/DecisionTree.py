import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        class_counts = np.bincount(y.astype(int))
        probabilities = class_counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def split_gini(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        y_left, y_right = y[left_mask], y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:  # split is invalid
            return float('inf')

        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)

        weighted_gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right
        return weighted_gini

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # finish criteria
        if depth == self.max_depth or num_samples < self.min_samples_split or num_classes == 1:
            return {'class': np.bincount(y.astype(int)).argmax()}

        # find best split
        best_split = None
        best_gini_value = float('inf')
        best_left_y, best_right_y = None, None
        for feature in range(num_features):
            thresholds = np.percentile(np.unique(X[:, feature]), np.linspace(10, 90, 10))
            for threshold in thresholds:
                gini_value = self.split_gini(X, y, feature, threshold)
                if gini_value < best_gini_value:
                    best_gini_value = gini_value
                    best_split = (feature, threshold)
                    best_left_y = y[X[:, feature] <= threshold]
                    best_right_y = y[X[:, feature] > threshold]

        if best_split is None:
            return {'class': np.bincount(y).argmax()}

        left = self._build_tree(X[X[:, best_split[0]] <= best_split[1]], best_left_y, depth + 1)
        right = self._build_tree(X[X[:, best_split[0]] > best_split[1]], best_right_y, depth + 1)

        return {
            'feature_index': best_split[0],
            'threshold': best_split[1],
            'left': left,
            'right': right
        }

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if 'class' in tree:
            return tree['class']

        # traverse tree based on treshold
        feature_index = tree['feature_index']
        threshold = tree['threshold']
        if sample[feature_index] <= threshold:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])