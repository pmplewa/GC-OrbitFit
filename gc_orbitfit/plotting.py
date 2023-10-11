import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from emcee.autocorr import integrated_time
from tqdm.auto import tqdm


def plot_data(data, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 2, **kwargs)

    for instrument, df in data.groupby("instrument"):
        df = df[["x", "y", "x_err", "y_err"]].dropna(axis="index")
        ax[0].errorbar(
            x=df["x"],
            y=df["y"],
            xerr=df["x_err"],
            yerr=df["y_err"],
            marker=".",
            linestyle="none",
            label=instrument,
        )

    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$x$ (arcsec)")
    ax[0].set_ylabel(r"$y$ (arcsec)")

    for instrument, df in data.groupby("instrument"):
        df = df[["vz", "vz_err"]].dropna(axis="index")
        ax[1].errorbar(
            x=df.index,
            y=df["vz"],
            yerr=df["vz_err"],
            marker=".",
            linestyle="none",
            label=instrument,
        )

    ax[1].set_ylabel(r"$v_z$ (km/s)")

    ax[1].legend(bbox_to_anchor=(1, 1))

    return ax[0].get_figure(), ax


def plot_resid(data, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(3, 1, sharex=True, **kwargs)

    for instrument, df in data.groupby("instrument"):
        df = df[["x", "x_err"]].dropna(axis="index")
        ax[0].errorbar(
            x=df.index,
            y=1e3 * df["x"],
            yerr=1e3 * df["x_err"],
            fmt=".",
            label=instrument,
        )

    for instrument, df in data.groupby("instrument"):
        df = df[["y", "y_err"]].dropna(axis="index")
        ax[1].errorbar(
            x=df.index,
            y=1e3 * df["y"],
            yerr=1e3 * df["y_err"],
            fmt=".",
            label=instrument,
        )

    for instrument, df in data.groupby("instrument"):
        df = df[["vz", "vz_err"]].dropna(axis="index")
        ax[2].errorbar(
            x=df.index,
            y=df["vz"],
            yerr=df["vz_err"],
            fmt=".",
            label=instrument,
        )

    ax[0].axhline(0, color="gray", linestyle="--")
    ax[1].axhline(0, color="gray", linestyle="--")
    ax[2].axhline(0, color="gray", linestyle="--")

    ax[0].set_ylabel(r"$\Delta x$ (mas)")
    ax[1].set_ylabel(r"$\Delta y$ (mas)")
    ax[2].set_ylabel(r"$\Delta v_z$ (km/s)")

    ax[0].legend(bbox_to_anchor=(1, 1))

    return ax[0].get_figure(), ax


def plot_resid_2d(data, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(**kwargs)

    for instrument, df in data.groupby("instrument"):
        ax.errorbar(
            x=1e3 * df["x"],
            y=1e3 * df["y"],
            xerr=1e3 * df["x_err"],
            yerr=1e3 * df["y_err"],
            fmt=".",
            alpha=0.5,
            label=instrument,
        )

    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")

    ax.set_aspect("equal")

    ax.set_xlabel(r"$\Delta x$ (mas)")
    ax.set_ylabel(r"$\Delta y$ (mas)")

    ax.legend()

    return ax.get_figure(), ax


def plot_trace(sampler, theta=None, show_burnin=True, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(sampler.ndim, sharex=True, **kwargs)

    if show_burnin:
        chain = sampler.get_chain()
    else:
        chain = sampler.get_chain(discard=sampler.nburn)

    for i in range(sampler.ndim):
        ax[i].plot(chain[:, :, i], color="k", alpha=0.2)
        ax[i].set_ylabel(sampler.names[i])
        if show_burnin:
            ax[i].axvline(sampler.nburn, color="gray", linestyle="-")
        if theta is not None:
            ax[i].axhline(theta[i], color="gray", linestyle="--")

    return ax[0].get_figure(), ax


def plot_corner(sampler, theta=None, corner_kwargs=None):
    samples = sampler.get_samples()

    default_opts = {"show_titles": True, "labels": sampler.names, "truths": theta}
    opts = {**default_opts, **(corner_kwargs or {})}
    return corner(samples, **opts)


def plot_acor(sampler, nmin=100, nsample=10, tol_length=50, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(**kwargs)

    chain = sampler.get_chain(discard=sampler.nburn)
    if len(chain) < nmin:
        raise ValueError("Not enough samples in chain.")

    n, tau = np.transpose(
        [
            (nmax, np.mean(integrated_time(chain[:nmax], tol=0)))
            for nmax in tqdm(
                np.linspace(nmin, len(chain), nsample, dtype=int),
                desc="Compute autocorrelation times",
            )
        ]
    )

    ax.plot(n, tau)
    ax.plot(n, n / tol_length, linestyle="--", color="gray")

    ax.set_ylabel(r"$\tau$")

    return ax.get_figure(), ax
