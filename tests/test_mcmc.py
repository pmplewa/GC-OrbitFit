from pathlib import Path
from typing import Any

from gc_orbitfit.data import load_sample_data
from gc_orbitfit.mcmc import MCMCSampler


def test_sample_orbit(tmp_path: Path, gp_noise_model: dict[str, Any]):
    data = load_sample_data(prefix="")

    chain_path = tmp_path / "chain.h5"
    sampler = MCMCSampler(
        data=data,
        names=gp_noise_model["names"],
        priors=gp_noise_model["priors"],
        model=gp_noise_model["model"],
        checkpoint=chain_path,
    )

    sampler.sample_orbit(2, processes=2)
    assert sampler.worker.iteration == 2

    assert chain_path.is_file()

    sampler.restore()

    sampler.sample_orbit(2, processes=2)
    assert sampler.worker.iteration == 4
