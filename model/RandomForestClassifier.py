import numpy as np
from . import DecisionTree


class RandomForest:
    def __init__(self, num_trees=100, min_samples_split=2, max_depth=None, random_state=0):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.trees = []

    def _bootstrap_sample(self, X, y, seed=None):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X), size=len(X), replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        for i in range(self.num_trees):
            seed = self.random_state + i if self.random_state is not None else None
            X_sample, y_sample = self._bootstrap_sample(X, y, seed)

            # create new decision tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            print(f"Fitting tree: {i}")
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # majority voting
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(preds).argmax() for preds in tree_preds.T])