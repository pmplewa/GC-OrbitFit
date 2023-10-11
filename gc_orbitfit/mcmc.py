import logging
from os import PathLike
from typing import Any, Sequence

import numpy as np
import pandas as pd
from emcee import EnsembleSampler
from emcee.backends import HDFBackend

# pylint: disable-next=no-name-in-module
from multiprocess import Pool

from .models import ModelType
from .priors import UniformPrior

logger = logging.getLogger("mcmc")


# pylint: disable-next=too-many-instance-attributes
class MCMCSampler:
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        data: pd.DataFrame,
        names: Sequence[str],
        priors: Sequence[UniformPrior],
        model: ModelType,
        nwalkers: int | None = None,
        checkpoint: str | PathLike | None = None,
    ):
        assert data.index.is_unique  # check for duplicate epochs
        assert len(names) > 0 and len(names) == len(priors)

        self._data = data
        self._names = names
        self._priors = priors
        self._model = model

        self._ndim = len(priors)

        if nwalkers is None:
            self._nwalkers = 2 * self.ndim  # minimum reasonable number
        else:
            self._nwalkers = nwalkers

        self._nburn = 0

        if checkpoint is not None:
            self._backend = HDFBackend(checkpoint)
        else:
            self._backend = None

        self._worker = None  # emcee sampler instance

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def names(self) -> Sequence[str]:
        return self._names

    @property
    def priors(self) -> Sequence[UniformPrior]:
        return self._priors

    @property
    def model(self) -> ModelType:
        return self._model

    @property
    def backend(self) -> HDFBackend | None:
        return self._backend

    @property
    def worker(self) -> EnsembleSampler:
        assert self._worker is not None
        return self._worker

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def nwalkers(self) -> int:
        return self._nwalkers

    @property
    def nburn(self) -> int:
        return self._nburn

    @nburn.setter
    def nburn(self, value):
        if value < 0:
            raise ValueError("'nburn' must be equal or greater 0.")

        nsamples, _, _ = self.get_chain().shape
        if value >= nsamples:
            raise ValueError("'nburn' must be smaller than 'nsamples'.")

        self._nburn = value

    def objective(self, theta: np.ndarray) -> float:
        """The Log-posterior probability density."""

        value = 0.0

        # evaluate the priors
        for i, prior in enumerate(self.priors):
            value += prior.log_pdf(theta[i])
            if not np.isfinite(value):
                return value  # early exit

        # evaluate the likelihood
        params = dict(zip(self.names, theta))
        value += self.model.log_likelihood(params, self.data)

        return value

    def sample_orbit(self, nsteps: int, **kwargs) -> None:
        self._worker = sample_orbit(self, nsteps, **kwargs)

    def restore(self, **kwargs) -> None:
        self._worker = sample_orbit(self, nsteps=None, **kwargs)

    def get_chain(self, **kwargs) -> Any:
        return self.worker.get_chain(**kwargs)

    def get_samples(self, **kwargs) -> Any:
        return self.get_chain(discard=self.nburn, flat=True, **kwargs)

    def get_sample(
        self, n: int | None = None, as_array: bool = False
    ) -> np.ndarray | dict[str, np.ndarray]:
        samples = self.get_samples()
        value = np.transpose(samples[np.random.randint(len(samples), size=n)])
        if as_array:
            return value
        return dict(zip(self.names, value))

    def get_mean(self, as_array: bool = False) -> np.ndarray | dict[str, np.ndarray]:
        samples = self.get_samples()
        value = np.mean(samples, axis=0)
        if as_array:
            return value
        return dict(zip(self.names, value))

    def get_std(self, as_array: bool = False) -> np.ndarray | dict[str, np.ndarray]:
        samples = self.get_samples()
        value = np.std(samples, axis=0)
        if as_array:
            return value
        return dict(zip(self.names, value))

    def get_autocorr_time(self, mean: bool = True, **kwargs) -> float:
        tau = self.worker.get_autocorr_time(discard=self.nburn, **kwargs)
        if mean:
            return np.mean(tau)
        return tau


def sample_orbit(
    sampler: MCMCSampler,
    nsteps: int | None = 0,
    theta0: np.ndarray | None = None,
    processes: int | None = None,
) -> EnsembleSampler:
    """Run the MCMC sampler.

    Note: For improved parallel performance, this function
    is not implemented as a class method of MCMCSampler.
    """
    # pylint: disable-next=not-callable
    with Pool(processes) as pool:
        worker = EnsembleSampler(
            sampler.nwalkers,
            sampler.ndim,
            sampler.objective,
            backend=sampler.backend,
            pool=pool,
        )

        if worker.backend.iteration == 0:
            logger.info("Starting new run")
            if theta0 is None:
                theta = np.array(
                    [
                        [prior.draw() for prior in sampler.priors]
                        for _ in range(sampler.nwalkers)
                    ]
                )
            else:
                theta = theta0
        else:
            logger.info("Resuming last run")
            # pylint: disable-next=protected-access
            theta = worker._previous_state
            assert theta is not None

        if nsteps is not None:
            if nsteps < 0:
                raise ValueError("'nsteps' must be equal or greater 0.")

            worker.run_mcmc(theta, nsteps, progress=True)
            logger.info("finished MCMC run")

        return worker
