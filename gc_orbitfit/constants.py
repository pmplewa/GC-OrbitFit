from astropy import constants, units

# mass, length and time units
M_UNIT = 1e6 * units.solMass
L_UNIT = units.arcsec.to(units.rad) * (1e3 * units.pc)  # * R0
T_UNIT = units.yr

# proper motion reference epoch
REF_TIME = 2000

# factor for converting to observed units
VELOCITY_CONVERSION_FACTOR_PER_R0 = float((L_UNIT / T_UNIT) / (units.km / units.s))

# constants (in the internal units)
G_TIMES_R03 = float(constants.G * M_UNIT * T_UNIT**2 / L_UNIT**3)
C_TIMES_R0 = float(constants.c * (T_UNIT / L_UNIT))
