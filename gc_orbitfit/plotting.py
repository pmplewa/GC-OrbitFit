import matplotlib.pyplot as plt


def plot_data(data, ax=None, **kwargs):  # noqa: ANN001, ANN201
    if ax is None:
        _, ax = plt.subplots(1, 2, **kwargs)

    for instrument, group in data.groupby("instrument"):
        df = group[["x", "y", "x_err", "y_err"]].dropna(axis="index")
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

    for instrument, group in data.groupby("instrument"):
        df = group[["vz", "vz_err"]].dropna(axis="index")
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


def plot_resid(data, ax=None, **kwargs):  # noqa: ANN001, ANN201
    if ax is None:
        _, ax = plt.subplots(3, 1, sharex=True, **kwargs)

    for instrument, group in data.groupby("instrument"):
        df = group[["x", "x_err"]].dropna(axis="index")
        ax[0].errorbar(
            x=df.index,
            y=1e3 * df["x"],
            yerr=1e3 * df["x_err"],
            fmt=".",
            label=instrument,
        )

    for instrument, group in data.groupby("instrument"):
        df = group[["y", "y_err"]].dropna(axis="index")
        ax[1].errorbar(
            x=df.index,
            y=1e3 * df["y"],
            yerr=1e3 * df["y_err"],
            fmt=".",
            label=instrument,
        )

    for instrument, group in data.groupby("instrument"):
        df = group[["vz", "vz_err"]].dropna(axis="index")
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
