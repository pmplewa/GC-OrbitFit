from pathlib import Path
from typing import Any

from gc_orbitfit.data import load_sample_data
from gc_orbitfit.sampling import NestedSampler


def test_sample_orbit(tmp_path: Path, gp_noise_model: dict[str, Any]) -> None:
    data = load_sample_data(prefix="")

    sampler = NestedSampler(
        data=data,
        names=gp_noise_model["names"],
        priors=gp_noise_model["priors"],
        model=gp_noise_model["model"],
    )

    checkpoint = tmp_path / "run.sav"
    sampler.sample_orbit(run_kwargs={"maxiter": 1}, checkpoint=str(checkpoint))
    assert checkpoint.is_file()
    assert sampler.worker.it > 0
