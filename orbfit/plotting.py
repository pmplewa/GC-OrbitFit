from corner import corner
from emcee.autocorr import integrated_time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .gp_tools import sample_gp


def plot_data(data, save=None, **kwargs):
    fig, ax = plt.subplots(1, 2, **kwargs)

    for instrument, df in data.groupby("instrument"):
        ax[0].errorbar(
            x=df["x"], y=df["y"],
            xerr=df["x_err"], yerr=df["y_err"],            
            marker=".", linestyle="none",
            label=instrument)
        
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$x$ (arcsec)")
    ax[0].set_ylabel(r"$y$ (arcsec)")

    for instrument, df in data.groupby("instrument"):
        ax[1].errorbar(
            x=df.index, y=df["vz"], yerr=df["vz_err"],
            marker=".", linestyle="none",
            label=instrument)   

    ax[1].set_ylabel(r"$v_z$ (km/s)")

    ax[1].legend(bbox_to_anchor=(1, 1))

    if save is not None:
        plt.savefig(save)

    return fig, ax

def plot_resid(data, save=None, **kwargs):
    fig, ax = plt.subplots(3, 1, sharex=True, **kwargs)

    for instrument, df in data.groupby("instrument"):
        ax[0].errorbar(
            x=df.index,
            y=1e3*df["x"],
            yerr=1e3*df["x_err"],            
            fmt=".",
            label=instrument)

    for instrument, df in data.groupby("instrument"):
        ax[1].errorbar(
            x=df.index,
            y=1e3*df["y"],
            yerr=1e3*df["y_err"],            
            fmt=".",
            label=instrument)  

    for instrument, df in data.groupby("instrument"):
        ax[2].errorbar(
            x=df.index,
            y=df["vz"],
            yerr=df["vz_err"],            
            fmt=".",
            label=instrument)

    ax[0].axhline(0, color="gray", linestyle="--")
    ax[1].axhline(0, color="gray", linestyle="--")
    ax[2].axhline(0, color="gray", linestyle="--")
        
    ax[0].set_ylabel(r"$\Delta x$ (mas)") 
    ax[1].set_ylabel(r"$\Delta y$ (mas)")     
    ax[2].set_ylabel(r"$\Delta v_z$ (km/s)")    

    ax[0].legend(bbox_to_anchor=(1, 1))     

    if save is not None:
        plt.savefig(save)

    return fig, ax

def plot_resid_2d(data, save=None, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    for instrument, df in data.groupby("instrument"):
        ax.errorbar(
            x=1e3*df["x"],
            y=1e3*df["y"],
            xerr=1e3*df["x_err"],
            yerr=1e3*df["y_err"],
            fmt=".",
            alpha=0.5,
            label=instrument)

    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")

    ax.set_aspect("equal")

    ax.set_xlabel(r"$\Delta x$ (mas)") 
    ax.set_ylabel(r"$\Delta y$ (mas)")

    ax.legend()

    if save is not None:
        plt.savefig(save)

    return fig, ax

def plot_trace(sampler, theta=None, show_burnin=True, save=None, **kwargs):
    fig, ax = plt.subplots(sampler.ndim, sharex=True, **kwargs)

    if show_burnin:
        chain = sampler.get_chain()
    else:
        chain = sampler.get_chain(discard=sampler.nburn)

    for i in range(sampler.ndim):
        ax[i].plot(chain[:,:,i], color="k", alpha=0.2)
        ax[i].set_ylabel(sampler.names[i])
        if show_burnin:
            ax[i].axvline(sampler.nburn, color="gray", linestyle="-")
        if theta is not None:
            ax[i].axhline(theta[i], color="gray", linestyle="--")

    if save is not None:
        plt.savefig(save)            

    return fig, ax

def plot_corner(sampler, theta=None, save=None, corner_kwargs={}):
    samples = sampler.get_samples()

    corner(samples,
        show_titles=True,
        labels=sampler.names,
        truths=(theta if theta is not None else None),
        **corner_kwargs)

    if save is not None:
        plt.savefig(save)    

    plt.show()

def plot_acor(sampler, nmin=100, nsample=10, tol=50, save=None, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    chain = sampler.get_chain(discard=sampler.nburn)
    assert len(chain) > nmin, "Not enough samples in chain"

    n, tau = np.transpose([(nmax, np.mean(integrated_time(chain[:nmax], tol=0)))
        for nmax in tqdm(np.linspace(100, len(chain), nsample, dtype=int),
                         desc="Computing autocorrelation times")])

    ax.plot(n, tau)
    ax.plot(n, n/50, linestyle="--", color="gray")

    ax.set_ylabel(r"$\tau$")

    if save is not None:
        plt.savefig(save)    

    return fig, ax
