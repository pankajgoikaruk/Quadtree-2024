import numpy as np

class Evaluation:
    def __init__(self, actual_values, predicted_values):
        self.actual_values = actual_values
        self.predicted_values = predicted_values

    def mean_absolute_error(self):
        return np.mean(np.abs(self.actual_values - self.predicted_values))

    def root_mean_squared_error(self):
        return np.sqrt(np.mean((self.actual_values - self.predicted_values) ** 2))

    def mean_absolute_percentage_error(self):
        return np.mean(np.abs((self.actual_values - self.predicted_values) / self.actual_values)) * 100

    def mean_error(self):
        return np.mean(self.actual_values - self.predicted_values)
