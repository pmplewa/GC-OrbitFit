from george import GP
from george.kernels import ExpSquaredKernel
import numpy as np
import pandas as pd

from .models import predict_resid, create_gp


def sample_gp(params, data, model, t_val, n=1):
    """Sample from a GP model for astrometric confusion"""

    resid = predict_resid(params, data, model)

    pos_resid = resid[["x", "x_err", "y", "y_err", "technique"]].dropna()

    gpx, gpy = create_gp(params)

    sample = pd.DataFrame(index=t_val)

    for technique, df in pos_resid.groupby("technique"):
        if technique == "interferometry":
            # unaffected by source confusion
            pass
        else:
            assert technique == "imaging"
            gpx.compute(df.index, df["x_err"])
            gpy.compute(df.index, df["y_err"])
            if n == 1:
                sample["x"] = gpx.sample_conditional(df["x"], t_val) 
                sample["y"] = gpy.sample_conditional(df["y"], t_val)
            else:
                # average over multiple realizations
                for i in range(n):
                    sample[f"x-{i}"] = gpx.sample_conditional(df["x"], t_val)
                    sample[f"y-{i}"] = gpy.sample_conditional(df["y"], t_val)
                sample["x"] = sample.filter(regex="x-").mean(axis=1)
                sample["y"] = sample.filter(regex="y-").mean(axis=1)

    return sample
