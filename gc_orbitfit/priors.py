import numpy as np


class UniformPrior:
    def __init__(self, a: float, b: float) -> None:
        assert a < b
        self.a = a  # lower bound
        self.b = b  # upper bound

    def transform(self, u: float) -> float:
        """Transform unit cube samples."""
        return self.a + (self.b - self.a) * u

    def log_pdf(self, x: float) -> float:
        """Return the log-probability density."""
        if (x < self.a) or (x > self.b):
            return -np.inf

        return -np.log(self.b - self.a)

    def bounds(self) -> tuple[float, float]:
        return self.a, self.b
