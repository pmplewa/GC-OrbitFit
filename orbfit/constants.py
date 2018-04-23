from astropy import constants, units


# mass, length and time units
m_unit = 1e6*units.solMass
l_unit = units.arcsec.to(units.rad)*(1e3*units.pc)
t_unit = units.yr

# proper motion reference epoch
reference_time = 2000

# internal velocity unit
v_unit = l_unit/t_unit

# factor for converting to observed units
velocity_conversion_factor=float(v_unit/(units.km/units.s))

# constants (in the internal units)
gravitational_constant = float(constants.G*m_unit*t_unit**2/l_unit**3)
speed_of_light = float(constants.c*t_unit/l_unit)
