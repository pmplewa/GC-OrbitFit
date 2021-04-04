import numpy as np


class UniformPrior():
    def __init__(self, a, b):
        assert a < b
        self.a = a # lower bound
        self.b = b # upper bound

    def log_pdf(self, x):
        """Return the log-probability density
        """
        if (x < self.a) or (x > self.b):
            return -np.inf
        return -np.log(self.b-self.a)

    def draw(self):
        """Return a random sample
        """
        return np.random.uniform(self.a, self.b)

    def bounds(self):
        return self.a, self.b
