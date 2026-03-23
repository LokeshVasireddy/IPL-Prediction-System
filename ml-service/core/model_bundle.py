import numpy as np

class IPLModelBundle:
    def __init__(
        self,
        model,
        dataset_version: str,
        feature_version: str
    ):
        self.model = model
        self.dataset_version = dataset_version
        self.feature_version = feature_version

    def preprocess(self, X: np.ndarray):
        """
        Expect already processed feature matrix
        shape: (n_samples, n_features)
        """

        if len(X.shape) != 2:
            raise ValueError("Expected 2D feature matrix")

        return X.reshape(X.shape[0], 1, X.shape[1])

    def predict(self, X: np.ndarray):
        """
        X should already be processed features
        """

        X = self.preprocess(X)
        preds = self.model.predict(X)

        return preds

    def info(self):
        return {
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version
        }