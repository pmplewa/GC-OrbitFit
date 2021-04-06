import numpy as np

from scipy.optimize import minimize, differential_evolution


class MaxLikeFitter():
    def __init__(self, data, names, bounds, model):
        assert data.index.is_unique # check for duplicate epochs
        assert len(names) == len(bounds)

        self._data = data
        self._names = names
        self._bounds = bounds
        self._model = model

        self._fit_result = None

    @property
    def data(self):
        return self._data

    @property
    def names(self):
        return self._names

    @property
    def bounds(self):
        return self._bounds

    @property
    def model(self):
        return self._model

    def objective(self, theta):
        """Negative log-likelihood function (to be minimized)
        """
        value = 0

        # evaluate likelihood
        params = dict(zip(self.names, theta))
        value += self.model.log_likelihood(params, self.data)

        return -value

    def fit_orbit(self):
        raise NotImplementedError

    def get_best_fit(self, as_array=False):
        assert self._fit_result is not None

        if as_array:
            return self._fit_result.x
        else:
            return dict(zip(self.names, self._fit_result.x))

class LBFGSBFitter(MaxLikeFitter):
    def fit_orbit(self, theta_init):
        """Find best-fit orbit parameters (using L-BFGS-B)
        """
        fit_result = minimize(self.objective, theta_init,
            bounds=self.bounds, method="L-BFGS-B", options=dict(disp=True, maxiter=np.inf))

        assert fit_result.success, "Fit was unsuccessful"
        self._fit_result = fit_result

class DifferentialEvolutionFitter(MaxLikeFitter):
    def fit_orbit(self):
        """Find best-fit orbit parameters (using differential evolution)
        """
        fit_result = differential_evolution(self.objective, self.bounds,
            maxiter=None, polish=True, disp=True)

        assert fit_result.success, "Fit was unsuccessful"
        self._fit_result = fit_result
