import numpy as np
from collections import Counter

class VotingClassifierScratch:
    def __init__(self, estimators, voting="hard"):
        """
        estimators: list of (name, model) tuples
        voting: "hard" for majority vote, "soft" for averaging probabilities
        """
        self.estimators = estimators
        self.voting = voting
        self.trained_models = []

    def fit(self, X, y):
        """Train all base estimators"""
        self.trained_models = []
        for _, model in self.estimators:
            model.fit(X, y)
            self.trained_models.append(model)

    def predict(self, X):
        """Predict based on the selected voting strategy"""
        if self.voting == "hard":
            predictions = np.array([model.predict(X) for model in self.trained_models])
            predictions = np.swapaxes(predictions, 0, 1)
            y_pred = [Counter(row).most_common(1)[0][0] for row in predictions]
            return np.array(y_pred)

        elif self.voting == "soft":
            probs = np.mean([model.predict_proba(X) for model in self.trained_models], axis=0)
            return np.argmax(probs, axis=1)
