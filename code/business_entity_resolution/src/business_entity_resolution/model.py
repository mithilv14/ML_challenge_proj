import math


def _sigmoid(value):
    if value >= 35:
        return 1.0
    if value <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


class LogisticRegressionSGD:
    """Small deterministic logistic model, implemented with the Python standard library."""

    def __init__(self, n_features, learning_rate=0.035, l2=0.0005, positive_weight=2.0):
        self.weights = [0.0] * n_features
        self.intercept = 0.0
        self.learning_rate = learning_rate
        self.l2 = l2
        self.positive_weight = positive_weight
        self.updates = 0

    def update(self, features, label):
        z = self.intercept + sum(w * x for w, x in zip(self.weights, features))
        probability = _sigmoid(z)
        weight = self.positive_weight if label else 1.0
        rate = self.learning_rate / math.sqrt(1.0 + self.updates / 250000.0)
        error = (float(label) - probability) * weight
        for i, value in enumerate(features):
            self.weights[i] += rate * (error * value - self.l2 * self.weights[i])
        self.intercept += rate * error
        self.updates += 1

    def predict_proba(self, features):
        z = self.intercept + sum(w * x for w, x in zip(self.weights, features))
        return _sigmoid(z)

    def to_dict(self):
        return {"weights": self.weights, "intercept": self.intercept, "updates": self.updates}

    @classmethod
    def from_dict(cls, payload):
        model = cls(len(payload["weights"]))
        model.weights = [float(value) for value in payload["weights"]]
        model.intercept = float(payload["intercept"])
        model.updates = int(payload.get("updates", 0))
        return model

