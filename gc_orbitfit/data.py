from os import PathLike
from typing import Optional, Union

import pandas as pd


def load_astrometry(
    path: Union[str, PathLike], instrument: str, rescale: Optional[float] = None
):
    data = pd.read_csv(path, index_col=0)
    data["instrument"] = instrument
    if rescale is not None:
        # rescale all error bars
        data[["x_err", "y_err"]] *= rescale
    if instrument == "SHARP":
        # speckle imaging
        data["technique"] = "imaging"
        # ... allow for preprocessing of the input data
        return data
    if instrument == "NACO":
        # adaptive optics imaging
        data["technique"] = "imaging"
        return data
    if instrument == "GRAVITY":
        # single field interferometry
        data["technique"] = "interferometry"
        return data
    raise ValueError(f"Unknown instrument '{instrument}'.")


def load_velocity(path: Union[str, PathLike], instrument: str):
    data = pd.read_csv(path, index_col=0)
    data["instrument"] = instrument
    if instrument == "NACO":
        # long slit spectroscopy
        data["technique"] = "spectroscopy"
        return data
    if instrument == "OSIRIS":
        # integral field spectroscopy
        data["technique"] = "spectroscopy"
        return data
    if instrument == "SINFONI":
        # integral field spectroscopy
        data["technique"] = "spectroscopy"
        return data
    raise ValueError(f"Unknown instrument '{instrument}'.")


def load_sample_data(
    prefix: str = "https://github.com/pmplewa/GC-OrbitFit/raw/main/",
) -> pd.DataFrame:
    """Load the default data set."""

    pos_data = pd.concat(
        [
            load_astrometry(f"{prefix}astrometry_SHARP.csv", instrument="SHARP"),
            load_astrometry(f"{prefix}astrometry_NACO.csv", instrument="NACO"),
        ]
    )

    vel_data = pd.concat(
        [
            load_velocity(f"{prefix}velocity_NACO.csv", instrument="NACO"),
            load_velocity(f"{prefix}velocity_OSIRIS.csv", instrument="OSIRIS"),
            load_velocity(f"{prefix}velocity_SINFONI.csv", instrument="SINFONI"),
        ]
    )

    assert len(pos_data) == 145, "Unexpected number of astrometric data points."
    assert len(vel_data) == 44, "Unexpected number of velocity data points."

    return pd.concat([pos_data, vel_data]).sort_index()
