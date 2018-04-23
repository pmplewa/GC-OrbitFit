# Stellar Orbit Fitting

Fit the orbit of the star S2 moving around the black hole at the center of the Milky Way:

*[Example Notebook](example.ipynb)*

[![](preview.png)](https://github.com/pmplewa/GC-OrbitFit/blob/master/example.ipynb)

This code takes into account first-order post-Newtonian corrections and gravitational redshift, as well as a time-correlated astrometric noise component. For the most part it is written with readability in mind, not performance.

To install the latest development version, run:

    pip install git+https://github.com/pmplewa/GC-OrbitFit.git

Resources:

* Data: [An Update on Monitoring Stellar Orbits in the Galactic Center](https://doi.org/10.3847/1538-4357/aa5c41)
* Noise model: [Unrecognized astrometric confusion in the Galactic Centre](https://dx.doi.org/10.1093/mnras/sty512)
* Orbit integration: [REBOUND](https://github.com/hannorein/rebound) and [REBOUNDx](https://github.com/dtamayo/reboundx)
