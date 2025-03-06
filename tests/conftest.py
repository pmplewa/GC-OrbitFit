from typing import Any

import numpy as np
import pytest

from gc_orbitfit.models import GPNoiseModel
from gc_orbitfit.priors import UniformPrior


@pytest.fixture
def gp_noise_model() -> dict[str, Any]:
    names, priors, theta0 = zip(
        *[
            ("M0", UniformPrior(3, 5), 4.449899),
            ("R0", UniformPrior(7, 9), 8.471232),
            ("a", UniformPrior(0.10, 0.15), 0.123663),
            ("e", UniformPrior(0.87, 0.90), 0.880469),
            ("inc", UniformPrior(2.3, 2.5), 2.362846),
            ("Omega", UniformPrior(3.9, 4.1), 3.97101),
            ("omega", UniformPrior(1.0, 1.2), 1.138985),
            ("tp", UniformPrior(2002.1, 2002.5), 2002.315867),
            ("x0", UniformPrior(-5e-3, 5e-3), 0.000712),
            ("y0", UniformPrior(-5e-3, 5e-3), -0.000223),
            ("vx0", UniformPrior(-1e-3, 1e-3), -0.000121),
            ("vy0", UniformPrior(-1e-3, 1e-3), 0.000044),
            ("vz0", UniformPrior(-50, 50), 8.812269),
            ("log_s2", UniformPrior(np.log(0.01e-6), np.log(1e-3)), -15.13448),
            ("log_taux", UniformPrior(np.log(0.01), np.log(5)), -0.630571),
            ("log_tauy", UniformPrior(np.log(0.01), np.log(5)), -2.151707),
        ],
        strict=True,
    )

    return {"model": GPNoiseModel(), "names": names, "priors": priors, "theta0": theta0}
