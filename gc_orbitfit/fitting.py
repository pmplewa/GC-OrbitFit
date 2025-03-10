from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .models import ModelType


class MaxLikeFitter(ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        names: Sequence[str],
        bounds: Sequence[tuple[float, float]],
        model: ModelType,
    ) -> None:
        assert data.index.is_unique  # check for duplicate epochs
        assert len(names) > 0
        assert len(names) == len(bounds)

        self._data = data
        self._names = names
        self._bounds = bounds
        self._model = model

        self._fit_result = None

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def names(self) -> Sequence[str]:
        return self._names

    @property
    def bounds(self) -> Sequence[tuple[float, float]]:
        return self._bounds

    @property
    def model(self) -> ModelType:
        return self._model

    def objective(self, theta: np.ndarray) -> float:
        """Evaluate the negative log-likelihood (to be minimized)."""
        params = dict(zip(self.names, theta, strict=True))
        return -self.model.log_likelihood(params, self.data)

    @abstractmethod
    def fit_orbit(self, theta0: np.ndarray | None = None) -> None: ...

    def get_best_fit(self, as_array: bool = False) -> np.ndarray | dict[str, float]:
        if self._fit_result is None:
            raise RuntimeError("No fit result available.")

        if as_array:
            return self._fit_result.x

        return dict(zip(self.names, self._fit_result.x, strict=True))


class LBFGSBFitter(MaxLikeFitter):
    def fit_orbit(self, theta0: np.ndarray | None = None) -> None:
        """Find the best-fit orbit parameters (using L-BFGS-B)."""
        if theta0 is None:
            raise ValueError("'theta0' may not be 'None'.")

        fit_result = minimize(
            self.objective,
            theta0,
            bounds=self.bounds,
            method="L-BFGS-B",
            options={"disp": True, "maxiter": np.inf},
        )

        if not fit_result.success:
            raise RuntimeError("The fit was unsuccessful.")

        self._fit_result = fit_result
