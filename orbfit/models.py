from george import GP
from george.kernels import ExpSquaredKernel
import numpy as np
import pandas as pd
import rebound
import reboundx
from scipy.optimize import fixed_point

from .constants import (gravitational_constant_times_R03,
    speed_of_light_times_R0, velocity_conversion_factor_per_R0, reference_time)


class REBOUNDModel():             
    def integrate_orbit(self, t_val, M0, R0, a, e, inc, Omega, omega, tp,
                        x0=0, y0=0, vx0=0, vy0=0, vz0=0, **effect_kwargs):
        # account for factor of R0 in l_unit
        gravitational_constant = gravitational_constant_times_R03/R0**3
        speed_of_light = speed_of_light_times_R0/R0
        velocity_conversion_factor = velocity_conversion_factor_per_R0*R0

        t_val = np.sort(t_val)

        sim = rebound.Simulation()
        rebx = reboundx.Extras(sim)      

        sim.integrator = "ias15"
        sim.G = gravitational_constant
        sim.t = t_val[0]

        # enable post-Newtonian corrections
        gr_effect = rebx.add("gr")
        gr_effect.params["c"] = speed_of_light          

        # add black hole particle
        sim.add(hash="black_hole", m=M0) # placed at the origin
        bh = sim.particles["black_hole"]
        bh.params["gr_source"] = 1
    
        # add star "test" particle
        sim.add(hash="test_particle", m=0,
            a=a, e=e, inc=inc, Omega=Omega, omega=omega, T=tp)
        p = sim.particles["test_particle"]

        if "Mp" and "rp" in effect_kwargs:
            # account for an extended mass distribution (Plummer profile)
            mass_effect = rebx.add_custom_force(central_force,
                force_is_velocity_dependent=False)   
            mass_effect.params["Mp"] = effect_kwargs["Mp"]
            mass_effect.params["rp"] = effect_kwargs["rp"]

        data = pd.DataFrame(index=t_val, columns=["t'", "x", "y", "vz"])

        for step, t_obs in enumerate(t_val):
            sim.integrate(t_obs, exact_finish_time=1) # time of observation

            # account for the light propagation delay (Roemer effect)
            t = t_obs - (p.z/speed_of_light * (1 - p.vz/speed_of_light))
            sim.integrate(t, exact_finish_time=1) # time of emission

            # convert to the coordinate system of the observations
            x_obs = -p.y
            y_obs = p.x

            # account for a possible drift of the astrometric reference frame
            x_obs += x0 + vx0 * (t - reference_time)
            y_obs += y0 + vy0 * (t - reference_time)            

            vz_obs = p.vz # the actual observable is the redshift (z = vz_obs/c)
            # account for the relativistic Doppler effect
            vz_obs += 0.5 * (p.vx**2 + p.vy**2 + p.vz**2) / speed_of_light
            # account for gravitational redshift
            vz_obs += sim.G*bh.m / np.sqrt(p.x**2 + p.y**2 + p.z**2) / speed_of_light
            # convert to observed units
            vz_obs *= velocity_conversion_factor
            # account for a possible radial velocity offset
            vz_obs += vz0

            data.loc[t_obs] = {"t'": t, "x": x_obs, "y": y_obs, "vz": vz_obs}

        return data

    def predict_data(self, params, t_val):
        return self.integrate_orbit(t_val, **params)

class GaussianNoise():
    def log_likelihood(self, params, data):
        """Gaussian log-likelihood function"""
         
        resid = predict_resid(params, data, self)
    
        pos_resid = resid[["x", "x_err", "y", "y_err"]].dropna()
        vel_resid = resid[["vz", "vz_err"]].dropna()
    
        value = 0

        value += log_gaussian_pdf(pos_resid["x"], pos_resid["x_err"])
        value += log_gaussian_pdf(pos_resid["y"], pos_resid["y_err"])
        value += log_gaussian_pdf(vel_resid["vz"], vel_resid["vz_err"])            
    
        return value
     
class GPNoise():    
    def log_likelihood(self, params, data):
        """GP log-likelihood function (to account for astrometric confusion)"""
    
        resid = predict_resid(params, data, self)
    
        pos_resid = resid[["x", "x_err", "y", "y_err", "technique"]].dropna()
        vel_resid = resid[["vz", "vz_err"]].dropna()    
    
        gpx, gpy = create_gp(params)
    
        value = 0
    
        for technique, df in pos_resid.groupby("technique"):
            if technique == "interferometry":
                # unaffected by source confusion
                value += log_gaussian_pdf(df["x"], df["x_err"])
                value += log_gaussian_pdf(df["y"], df["y_err"])
            else:
                assert technique == "imaging"
                gpx.compute(df.index, df["x_err"])
                gpy.compute(df.index, df["y_err"])
                value += gpx.lnlikelihood(df["x"])
                value += gpy.lnlikelihood(df["y"])
    
        value += log_gaussian_pdf(vel_resid["vz"], vel_resid["vz_err"])

        return value

def log_gaussian_pdf(r, sigma):
    return -0.5*np.sum(r**2/sigma**2) -0.5*np.sum(np.log(2*np.pi*sigma**2))

def predict_resid(params, data, model):
    pred = []

    for technique, df in data.groupby("technique"):
        if technique == "interferometry":
            # tie interferometric data points directly to the mass
            params_copy = params.copy()
            params_copy["x0"] = 0
            params_copy["y0"] = 0
            params_copy["vx0"] = 0
            params_copy["vy0"] = 0
            pred.append(model.predict_data(params_copy, df.index))
        else:
            assert (technique == "imaging") or (technique == "spectroscopy")
            pred.append(model.predict_data(params, df.index))

    pred = pd.concat(pred)

    return pd.concat([
        pred["t'"],
        data["x"] - pred["x"],
        data["y"] - pred["y"],
        data["x_err"],
        data["y_err"],
        data["vz"] - pred["vz"],
        data["vz_err"],
        data["instrument"],
        data["technique"]],
        axis=1)

def create_gp(params):
    # GP parameters
    s2 = np.exp(params["log_s2"])
    taux = np.exp(params["log_taux"])
    tauy = np.exp(params["log_tauy"])

    gpx = GP(s2*ExpSquaredKernel(taux))
    gpy = GP(s2*ExpSquaredKernel(tauy))

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
            ar = sim.G*Mp * r/(r**2 + rp**2)**(3/2)
            p.ax -= ar * p.x/r
            p.ay -= ar * p.y/r
            p.ax -= ar * p.x/r 

"""
As an alternative to the approximate solution used, the time retardation
equation can also be solved iteratively to account for the light propagation
delay:

def retardation_equation(t, t_obs, sim, particle_hash):
    sim.integrate(t)
    p = sim.particles[particle_hash]
    return t_obs - p.z/speed_of_light

t = fixed_point(retardation_equation, t_obs,
                args=(t_obs, sim, "test_particle"),
                xtol=1e-8)
"""
