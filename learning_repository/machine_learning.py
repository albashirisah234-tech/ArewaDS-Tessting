"""Introductory machine learning examples."""

from dataclasses import dataclass


@dataclass
class SimpleLinearRegression:
    """A tiny linear regression model for learning purposes."""

    slope: float = 0.0
    intercept: float = 0.0

    def fit(self, x_values: list[float], y_values: list[float]) -> None:
        """Fit a 1D linear regression model using least squares."""
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same length")
        if len(x_values) < 2:
            raise ValueError("at least two data points are required")

        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        if denominator == 0:
            raise ValueError("x_values must not all be the same")

        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, x_value: float) -> float:
        """Predict a y value from a single x value."""
        return self.slope * x_value + self.intercept

