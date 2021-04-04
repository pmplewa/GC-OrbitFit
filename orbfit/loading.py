import pandas as pd


def load_astrometry(path, instrument, rescale=None):
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
    elif instrument == "NACO":
        # adaptive optics imaging
        data["technique"] = "imaging"
        return data
    elif instrument == "GRAVITY":
        # single field interferometry
        data["technique"] = "interferometry"
        return data
    else:
        raise Exception("Unknown instrument")

def load_velocity(path, instrument):
    data = pd.read_csv(path, index_col=0)
    data["instrument"] = instrument
    if instrument == "NACO":
        # long slit spectroscopy
        data["technique"] = "spectroscopy"
        return data
    elif instrument == "OSIRIS":
        # integral field spectroscopy
        data["technique"] = "spectroscopy"
        return data
    elif instrument == "SINFONI":
        # integral field spectroscopy
        data["technique"] = "spectroscopy"
        return data
    else:
        raise Exception("Unknown instrument")

def load_default():
    """Load the default data set"""

    pos_data = pd.concat([
        load_astrometry("astrometry_SHARP.csv", instrument="SHARP"),
        load_astrometry("astrometry_NACO.csv", instrument="NACO")])

    vel_data = pd.concat([
        load_velocity("velocity_NACO.csv", instrument="NACO"),
        load_velocity("velocity_OSIRIS.csv", instrument="OSIRIS"),
        load_velocity("velocity_SINFONI.csv", instrument="SINFONI")])

    assert len(pos_data) == 145, "Unexpected number of astrometric data points"
    assert len(vel_data) == 44, "Unexpected number of velocity data points"

    return pd.concat([pos_data, vel_data]).sort_index()
