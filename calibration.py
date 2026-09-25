"""
Probability calibration methods for the Ask-or-Act intent predictor.

Kept in their own module so that models saved by train_predictor.py can be
loaded by the robot's controller.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """Divides the model's logit by a single temperature T fitted to minimise log loss."""
    def fit(self, p, y):
        z = logit(p)
        res = minimize_scalar(lambda lt: log_loss(y, np.clip(sigmoid(z / np.exp(lt)), EPS, 1 - EPS)),
                              bounds=(-3, 3), method='bounded')
        self.T = float(np.exp(res.x))
        return self

    def transform(self, p):
        return sigmoid(logit(p) / self.T)


class IsotonicCalibrator:
    """Monotonic, piecewise-constant mapping from raw to calibrated probability."""
    def fit(self, p, y):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(p, y)
        return self

    def transform(self, p):
        return np.clip(self.iso.predict(p), EPS, 1 - EPS)
