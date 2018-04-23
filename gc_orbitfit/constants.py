# pylint: disable=no-member

from astropy import constants, units

# mass, length and time units
m_unit = 1e6 * units.solMass
l_unit = units.arcsec.to(units.rad) * (1e3 * units.pc)  # * R0
t_unit = units.yr

# proper motion reference epoch
ref_time = 2000

# factor for converting to observed units
velocity_conversion_factor_per_R0 = float((l_unit / t_unit) / (units.km / units.s))

# constants (in the internal units)
G_times_R03 = float(constants.G * m_unit * t_unit**2 / l_unit**3)
c_times_R0 = float(constants.c * (t_unit / l_unit))
