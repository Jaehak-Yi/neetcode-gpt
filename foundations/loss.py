import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        min_y = 10 ** -7
        y_pred = np.clip(y_pred, min_y, 1-min_y)
        log_y = np.log(y_pred)
        log_1y = np.log(1-y_pred)

        temp = y_true * log_y + (1-y_true) * log_1y
        return np.round((-1/n) * np.sum(temp), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        min_y = 10 ** -7
        y_pred = np.clip(y_pred, min_y, 1-min_y)
        log_y = np.log(y_pred)

        temp = y_true * log_y
        return np.round((-1/n) * np.sum(temp), 4)
