import logging
import numpy as np

from emcee import EnsembleSampler
from emcee.backends import HDFBackend
from multiprocess import Pool


logger = logging.getLogger("mcmc")

class MCMCSampler():
    def __init__(self, data, names, priors, model, nwalkers=None, nburn=0, checkpoint=None):
        assert data.index.is_unique # check for duplicate epochs
        assert len(names) == len(priors)

        self._data = data
        self._names = names
        self._priors = priors
        self._model = model

        self._ndim = len(priors)

        if nwalkers is None:
            self._nwalkers = 2 * self.ndim # minimum reasonable number
        else:
            self._nwalkers = nwalkers

        self._nburn = 0

        if checkpoint is not None:
            self._backend = HDFBackend(checkpoint)
        else:
            self._backend = None

        self._worker = None # emcee sampler instance

    @property
    def data(self):
        return self._data

    @property
    def names(self):
        return self._names

    @property
    def priors(self):
        return self._priors

    @property
    def model(self):
        return self._model

    @property
    def backend(self):
        return self._backend

    @property
    def worker(self):
        return self._worker

    @property
    def ndim(self):
        return self._ndim

    @property
    def nwalkers(self):
        return self._nwalkers

    @property
    def nburn(self):
        return self._nburn

    @nburn.setter
    def nburn(self, value):
        assert value >= 0
        nsamples, _, _ = self.get_chain().shape
        assert value < nsamples
        self._nburn = value

    def objective(self, theta):
        """Log-posterior probability density
        """
        value = 0

        # evaluate priors
        for i, prior in enumerate(self.priors):
            value += prior.log_pdf(theta[i])
            if not np.isfinite(value):
                return value # early exit

        # evaluate likelihood
        params = dict(zip(self.names, theta))
        value += self.model.log_likelihood(params, self.data)

        return value

    def sample_orbit(self, nsteps, **kwargs):
        self._worker = sample_orbit(self, nsteps, **kwargs)

    def restore(self, **kwargs):
        self._worker = sample_orbit(self, nsteps=0, **kwargs)

    def get_chain(self, **kwargs):
        return self.worker.get_chain(**kwargs)

    def get_samples(self, **kwargs):
        return self.get_chain(discard=self.nburn, flat=True, **kwargs)

    def get_sample(self, n=None, as_array=False):
        samples = self.get_samples()
        value = np.transpose(samples[np.random.randint(len(samples), size=n)])
        if as_array:
            return value
        else:
            return dict(zip(self.names, value))

    def get_mean(self, as_array=False):
        samples = self.get_samples()
        value = np.mean(samples, axis=0)
        if as_array:
            return value
        else:
            return dict(zip(self.names, value))

    def get_std(self, as_array=False):
        samples = self.get_samples()
        value = np.std(samples, axis=0)
        if as_array:
            return value
        else:
            return dict(zip(self.names, value))

    def get_autocorr_time(self, mean=True, **kwargs):
        tau = self.worker.get_autocorr_time(discard=self.nburn, **kwargs)
        if mean:
            return np.mean(tau)
        else:
            return tau

def sample_orbit(sampler, nsteps, resume=False, processes=None):
    """Run the MCMC sampler

    Note: For improved parallel performance this function is not implemented as
    a class method of MCMCSampler.
    """
    assert nsteps >= 0

    with Pool(processes) as pool:
        worker = EnsembleSampler(sampler.nwalkers, sampler.ndim, sampler.objective,
            backend=sampler.backend, pool=pool)

        if nsteps > 0:
            if resume:
                logger.info("Resuming last run")
                theta = None # resume sampling from last position
            else:
                logger.info("Starting new run")
                theta = np.array([[prior.draw() for prior in sampler.priors]
                    for n in range(sampler.nwalkers)])

            worker.run_mcmc(theta, nsteps, progress=True)

        else:
            logger.info("Restoring from checkpoint")
            assert sampler.backend is not None

        return worker
