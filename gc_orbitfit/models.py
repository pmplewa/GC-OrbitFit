from typing import Union

import numpy as np
import pandas as pd
import rebound
import reboundx
from george import GP
from george.kernels import ExpSquaredKernel

from .constants import (
    G_times_R03,
    c_times_R0,
    ref_time,
    velocity_conversion_factor_per_R0,
)


class REBOUNDModel:
    @staticmethod
    # pylint: disable-next=too-many-arguments,too-many-locals
    def integrate_orbit(
        t_val: np.ndarray,
        M0: float,
        R0: float,
        a: float,
        e: float,
        inc: float,
        Omega: float,
        omega: float,
        tp: float,
        x0: float = 0.0,
        y0: float = 0.0,
        vx0: float = 0.0,
        vy0: float = 0.0,
        vz0: float = 0.0,
        **effect_kwargs
    ) -> pd.DataFrame:
        # account for factor of R0 in l_unit
        G = G_times_R03 / R0**3
        c = c_times_R0 / R0
        velocity_conversion_factor = velocity_conversion_factor_per_R0 * R0

        t_val = np.sort(t_val)

        sim = rebound.Simulation()
        rebx = reboundx.Extras(sim)

        sim.integrator = "ias15"
        sim.G = G
        sim.t = t_val[0]

        # enable post-Newtonian corrections
        gr = rebx.load_force("gr")
        rebx.add_force(gr)
        gr.params["c"] = c

        # add black hole particle
        sim.add(hash="black_hole", m=M0)  # placed at the origin
        bh = sim.particles["black_hole"]
        bh.params["gr_source"] = 1

        # add star "test" particle
        orbit = {"a": a, "e": e, "inc": inc, "Omega": Omega, "omega": omega, "T": tp}
        sim.add(hash="test_particle", m=0, **orbit)
        p = sim.particles["test_particle"]

        if "Mp" in effect_kwargs and "rp" in effect_kwargs:
            # account for an extended mass distribution (Plummer profile)
            mass_effect = rebx.add_custom_force(
                central_force, force_is_velocity_dependent=False
            )
            mass_effect.params["Mp"] = effect_kwargs["Mp"]
            mass_effect.params["rp"] = effect_kwargs["rp"]

        data = pd.DataFrame(index=t_val, columns=["t'", "x", "y", "vz"])
        for t_obs in t_val:
            sim.integrate(t_obs, exact_finish_time=1)  # time of observation

            # account for the light propagation delay (Roemer effect)
            t = t_obs - (p.z / c * (1 - p.vz / c))
            sim.integrate(t, exact_finish_time=1)  # time of emission

            # convert to the coordinate system of the observations
            x_obs = -p.y
            y_obs = p.x

            # account for a possible drift of the astrometric reference frame
            x_obs += x0 + vx0 * (t - ref_time)
            y_obs += y0 + vy0 * (t - ref_time)

            # account for the relativistic Doppler effect
            beta_costheta = p.vz / c
            beta2 = (p.vx**2 + p.vy**2 + p.vz**2) / c**2
            zD = (1 + beta_costheta) / np.sqrt(1 - beta2) - 1

            # account for the gravitational redshift
            rs = 2 * sim.G * bh.m / c**2
            zG = 1 / np.sqrt(1 - rs / np.sqrt(p.x**2 + p.y**2 + p.z**2)) - 1

            # calculate the measured radial velocity
            vz_obs = (zD + zG) * c

            # convert to observed units
            vz_obs *= velocity_conversion_factor

            # account for a possible radial velocity offset
            vz_obs += vz0

            data.loc[t_obs] = {"t'": t, "x": x_obs, "y": y_obs, "vz": vz_obs}

        return data

    def predict_data(self, params: dict[str, float], t_val: np.ndarray) -> pd.DataFrame:
        return self.integrate_orbit(t_val, **params)


# pylint: disable-next=too-few-public-methods
class GaussianNoise:
    def log_likelihood(self, params: dict[str, float], data: pd.DataFrame) -> float:
        """The Gaussian log-likelihood function."""

        resid = predict_resid(params, data, self)  # type: ignore

        pos_resid = resid[["x", "x_err", "y", "y_err"]].dropna()
        vel_resid = resid[["vz", "vz_err"]].dropna()

        value = 0.0

        value += log_gaussian_pdf(pos_resid["x"], pos_resid["x_err"])
        value += log_gaussian_pdf(pos_resid["y"], pos_resid["y_err"])
        value += log_gaussian_pdf(vel_resid["vz"], vel_resid["vz_err"])

        return value


# pylint: disable-next=too-few-public-methods
class GPNoise:
    def log_likelihood(self, params: dict[str, float], data: pd.DataFrame) -> float:
        """The GP log-likelihood function (to account for astrometric confusion)."""

        resid = predict_resid(params, data, self)  # type: ignore

        pos_resid = resid[["x", "x_err", "y", "y_err", "technique"]].dropna()
        vel_resid = resid[["vz", "vz_err"]].dropna()

        gpx, gpy = create_gp(params)

        value = 0.0

        for technique, df in pos_resid.groupby("technique"):
            if technique == "interferometry":
                # unaffected by source confusion
                value += log_gaussian_pdf(df["x"], df["x_err"])
                value += log_gaussian_pdf(df["y"], df["y_err"])
            else:
                assert technique == "imaging"
                gpx.compute(df.index, df["x_err"])
                gpy.compute(df.index, df["y_err"])
                value += gpx.log_likelihood(df["x"])
                value += gpy.log_likelihood(df["y"])

        value += log_gaussian_pdf(vel_resid["vz"], vel_resid["vz_err"])

        return value


class GaussianNoiseModel(REBOUNDModel, GaussianNoise):
    pass


class GPNoiseModel(REBOUNDModel, GPNoise):
    pass


ModelType = Union[GaussianNoiseModel, GPNoiseModel]


def log_gaussian_pdf(r: np.ndarray, sigma: np.ndarray) -> float:
    return -0.5 * np.sum(r**2 / sigma**2) - 0.5 * np.sum(
        np.log(2 * np.pi * sigma**2)
    )


def predict_resid(params: dict[str, float], data: pd.DataFrame, model: ModelType):
    data_tmp = []

    for technique, df in data.groupby("technique"):
        if technique == "interferometry":
            # tie interferometric data points directly to the mass
            params_copy = params.copy()
            params_copy["x0"] = 0
            params_copy["y0"] = 0
            params_copy["vx0"] = 0
            params_copy["vy0"] = 0
            data_tmp.append(model.predict_data(params_copy, df.index))
        else:
            assert technique in ("imaging", "spectroscopy")
            data_tmp.append(model.predict_data(params, df.index))

    data_pred = pd.concat(data_tmp)

    return pd.concat(
        [
            data_pred["t'"],
            data["x"] - data_pred["x"],
            data["y"] - data_pred["y"],
            data["x_err"],
            data["y_err"],
            data["vz"] - data_pred["vz"],
            data["vz_err"],
            data["instrument"],
            data["technique"],
        ],
        axis=1,
    )


def create_gp(params: dict[str, float]):
    # GP parameters
    s2 = np.exp(params["log_s2"])
    taux = np.exp(params["log_taux"])
    tauy = np.exp(params["log_tauy"])

    gpx = GP(s2 * ExpSquaredKernel(taux))
    gpy = GP(s2 * ExpSquaredKernel(tauy))

    return gpx, gpy


def central_force(reb_sim, rebx_effect, particles, nparticles):
    sim = reb_sim.contents
    effect = rebx_effect.contents

    # effect parameters
    Mp = effect.params["Mp"]
    rp = effect.params["rp"]

    for i in range(nparticles):
        p = particles[i]
        # only act on test particles
        if p.m == 0:
            r = np.sqrt(p.x**2 + p.y**2 + p.z**2)
            ar = sim.G * Mp * r / (r**2 + rp**2) ** (3 / 2)
            p.ax -= ar * p.x / r
            p.ay -= ar * p.y / r
            p.ax -= ar * p.x / r


"""
As an alternative to the approximate solution used, the time retardation
equation can also be solved iteratively to account for the light propagation
delay:

def retardation_equation(t, t_obs, sim, particle_hash):
    sim.integrate(t)
    p = sim.particles[particle_hash]
    return t_obs - p.z / speed_of_light

t = fixed_point(retardation_equation, t_obs, args=(t_obs, sim, "test_particle"), xtol=1e-8)
"""
