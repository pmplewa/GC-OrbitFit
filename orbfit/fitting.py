import numpy as np
from scipy.optimize import minimize


class MaxLikeFitter():
    def __init__(self, data, names, bounds, model):
        assert data.index.is_unique # check for duplicate epochs
        assert len(names) == len(bounds)

        self.data = data
        self.names = names
        self.bounds = bounds
        self.model = model

        self.fit_result = None

    def objective(self, theta):
        """Negative log-likelihood function (to be minimized)"""
    
        value = 0
            
        # evaluate likelihood
        params = dict(zip(self.names, theta))
        value += self.model.log_likelihood(params, self.data)
        
        return -value

    def fit_orbit(self, theta_init):
        """Find best-fit orbit parameters"""
    
        fit_result = minimize(self.objective,
                              theta_init,
                              bounds=self.bounds,
                              method="L-BFGS-B",
                              options={"disp": True,
                                       "maxiter": np.inf})
    
        assert fit_result.success, "Fit was unsuccessful"
        self.fit_result = fit_result

    def get_best_fit(self, as_array=False):
        if as_array:
            return self.fit_result.x
        else:
            return dict(zip(self.names, self.fit_result.x))
