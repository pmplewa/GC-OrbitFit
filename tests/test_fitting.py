from typing import Any

from gc_orbitfit.data import load_sample_data
from gc_orbitfit.fitting import LBFGSBFitter


def test_fit_orbit(gp_noise_model: dict[str, Any]):
    data = load_sample_data(prefix="")

    fitter = LBFGSBFitter(
        data=data,
        names=gp_noise_model["names"],
        bounds=[prior.bounds() for prior in gp_noise_model["priors"]],
        model=gp_noise_model["model"],
    )

    fitter.fit_orbit(gp_noise_model["theta0"])
    assert fitter.get_best_fit()
