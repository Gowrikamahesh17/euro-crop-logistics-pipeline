"""Model wrapper classes shared by train.py and evaluate.py.

Kept in its own module (rather than inside train.py) so its __module__ stays
stable for joblib pickling regardless of whether train.py or evaluate.py is
the script being run directly as __main__.
"""

from sklearn.preprocessing import LabelEncoder


class LabelEncodedClassifier:
    """Wraps a classifier that requires integer labels (e.g. XGBoost) so it can
    be fit/predict directly on string class labels like the other models."""

    def __init__(self, model):
        self.model = model
        self.encoder = LabelEncoder()

    def fit(self, X, y):
        y_enc = self.encoder.fit_transform(y)
        self.model.fit(X, y_enc)
        return self

    def predict(self, X):
        return self.encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)
