import os
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import dynesty
import numpy as np
import pandas as pd
from dynesty.pool import Pool

from .models import ModelType
from .priors import UniformPrior


class HasDimensions(Protocol):
    ndim: int


T = TypeVar("T", bound=HasDimensions)


class Sampler(Generic[T]):
    def __init__(
        self,
        data: pd.DataFrame,
        names: Sequence[str],
        priors: Sequence[UniformPrior],
        model: ModelType,
    ) -> None:
        assert data.index.is_unique  # check for duplicate epochs
        assert len(names) > 0
        assert len(names) == len(priors)

        self._data = data
        self._names = names
        self._priors = priors
        self._model = model

        self._ndim = len(priors)
        self._worker: T | None = None

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
    def ndim(self) -> int:
        return self._ndim

    @property
    def worker(self) -> T:
        if self._worker is None:
            raise RuntimeError("No worker available.")
        return self._worker

    @worker.setter
    def worker(self, worker: T) -> None:
        if worker.ndim != self.ndim:
            raise ValueError("Incompatible number of dimensions.")
        self._worker = worker

    def objective(self, theta: np.ndarray) -> float:
        """Call the log-likelihood function."""
        params = dict(zip(self.names, theta, strict=True))
        return self.model.log_likelihood(params, self.data)

    @abstractmethod
    def sample_orbit(self, **kwargs) -> T: ...


class NestedSampler(Sampler[dynesty.NestedSampler]):
    def prior_transform(self, phi: np.ndarray) -> np.ndarray:
        """Apply the prior transform."""
        return np.array(
            [prior.transform(x) for prior, x in zip(self.priors, phi, strict=True)],
        )

    def sample_orbit(self, **kwargs) -> dynesty.NestedSampler:
        self.worker = sample_orbit(self, **kwargs)
        return self.worker

    def restore(self, checkpoint: str, **kwargs) -> dynesty.NestedSampler:
        self.worker = dynesty.NestedSampler.restore(checkpoint, **kwargs)
        return self.worker


def sample_orbit(
    sampler: NestedSampler,
    *,
    num_jobs: int | None = None,
    sampler_kwargs: dict[str, Any] | None = None,
    run_kwargs: dict[str, Any] | None = None,
    checkpoint: str | None = None,
) -> dynesty.NestedSampler:
    """Run the MCMC sampler.

    Note: For improved parallel performance, this function
    is not implemented as a class method of MCMCSampler.
    """
    sampler_kwargs = sampler_kwargs or {}
    run_kwargs = run_kwargs or {}
    with Pool(
        num_jobs or os.cpu_count(),
        sampler.objective,
        sampler.prior_transform,
    ) as pool:
        if checkpoint and Path(checkpoint).is_file():
            worker = sampler.restore(checkpoint, pool=pool)
        else:
            worker = dynesty.NestedSampler(
                loglikelihood=pool.loglike,
                prior_transform=pool.prior_transform,
                ndim=sampler.ndim,
                pool=pool,
                **sampler_kwargs,
            )

        checkpoint = run_kwargs.pop("checkpoint_file", checkpoint)
        worker.run_nested(
            checkpoint_file=checkpoint, resume=checkpoint is not None, **run_kwargs
        )

    return worker
