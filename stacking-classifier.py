import numpy as np
from copy import deepcopy

class StackingClassifierScratch:
    def __init__(self, base_estimators, meta_estimator):
        """
        base_estimators: list of model instances (must have fit() and predict())
        meta_estimator: model instance trained on base estimator predictions
        """
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.trained_base = []

    def fit(self, X, y):
        """Train base models and then the meta-model"""
        self.trained_base = []
        base_predictions = []

        for model in self.base_estimators:
            cloned_model = deepcopy(model)
            cloned_model.fit(X, y)
            preds = cloned_model.predict(X)
            base_predictions.append(preds)
            self.trained_base.append(cloned_model)

        # Stack predictions as new feature set
        meta_X = np.column_stack(base_predictions)
        self.meta_estimator.fit(meta_X, y)

    def predict(self, X):
        """Predict using trained base and meta models"""
        base_preds = [model.predict(X) for model in self.trained_base]
        meta_X = np.column_stack(base_preds)
        return self.meta_estimator.predict(meta_X)
