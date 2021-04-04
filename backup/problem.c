#include "multinest.h"
#include "rebound.h"
#include "reboundx.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// unit-dependent global variables
double gravitational_constant_times_R03, speed_of_light_times_R0;
double velocity_conversion_factor_per_R0;

// main data structure
struct data {
  int N_points;
  struct point* points;
};

// data point
struct point {
  double t; // time of observation
  double x; // angular offset along RA
  double x_err;
  double y; // angular offset along DEC
  double y_err;
  double vz; // radial velocity
  double vz_err;
  int type;
};

// data types
#define position_type 1
#define velocity_type 2

// data import function
struct data* import_data(char* filename, int N_points);

// log-likelihood function
void loglike(double *cube, int *n_dims, int *n_pars, double *new_value, void *context);

// uniform prior transformation
double prior_uniform(double x, double a, double b);

// callback function
void dumper(int *nsamples, int *n_live, int *n_pars, double **physLive, double **posterior,
            double **paramConstr, double *maxloglike, double *logZ, double *INSlogZ,
            double *logZerr, void *context);

int main(int argc, char* argv[]) {
  // import data
  struct data* d = import_data("data.csv", 189);

  // set MultiNest parameters
  int flag_imp = 1; // enable importance sampling
  int flag_mode_sep = 1; // enable mode separation
  int flag_const_eff = 0; // enable constant efficiency mode
  int n_live = 500; // number of live points
  double eff = 0.8; // required efficiency
  double tol = 0.5; // stopping criterion
  int n_dims = 13; // dimensionality (number of free parameters)
  int n_pars = 13; // total number of parameters (including derived parameters)
  int n_sep = 13; // number of parameters to do mode separation on
  int update_interval = 10; // update output files
  double Ztol = -1E90; // ignore modes with logZ < Ztol
  int max_modes = 100; // expected max. number of modes
  int flag_wrap[n_dims]; // enable periodic boundary conditions
  for(int i = 0; i < n_dims; i++) flag_wrap[i] = 0;
  char root[100] = "chains/S2-"; // root name for output files
  int seed = -1; // random number generator seed
  int flag_feedback = 1; // enable feedback on standard output
  int flag_resume = 1; // resume from a previous job
  int flag_outfile = 1; // write output files
  int flag_mpi = 1; // initialize MPI routines (if compiling with MPI)
  double log_zero = -DBL_MAX; // ignore points with log_like < log_zero
  int max_iter = 0; // max. number of iterations

  // define unit system
  double m_unit = 1.988475415338144e+39; // 1e6 solar masses in g
  double l_unit = 1.4959787069882792e+16; // 1 as in cm at 1 kpc (* R0)
  double t_unit = 31557600.; // 1 yr in s

  velocity_conversion_factor_per_R0 = (l_unit/t_unit)/100000.; // for converting to km/s

  // define natural constants (in the internal units)
  gravitational_constant_times_R03 = 6.67408E-8 * m_unit*pow(t_unit, 2)/pow(l_unit, 3);
  speed_of_light_times_R0 = 29979245800. * t_unit/l_unit;

  // run MultiNest
  run(flag_imp, flag_mode_sep, flag_const_eff, n_live, tol, eff, n_dims, n_pars,
      n_sep, max_modes, update_interval, Ztol, root, seed, flag_wrap,
      flag_feedback, flag_resume, flag_outfile, flag_mpi, log_zero, max_iter,
      loglike, dumper, d);
}

void loglike(double *cube, int *n_dims, int *n_pars, double *new_value, void *context) {
  struct data* d = context;
  struct point* points = d->points;

  // apply prior transforms
  double M0 = prior_uniform(cube[0], 3., 5.); // black hole mass
  double R0 = prior_uniform(cube[1], 7., 9.); // black hole distance

  // account for factor of R0 in l_unit
  double gravitational_constant = gravitational_constant_times_R03/pow(R0, 3);
  double speed_of_light = speed_of_light_times_R0/R0;
  double velocity_conversion_factor = velocity_conversion_factor_per_R0*R0;

  double a = prior_uniform(cube[2], 0.10, 0.15); // semi-major axis
  double e = prior_uniform(cube[3], 0.87, 0.90); // eccentricity
  double inc = prior_uniform(cube[4], 2.3, 2.5); // inclination
  double Omega = prior_uniform(cube[5], 3.9, 4.1); // longiture of asc. node
  double omega = prior_uniform(cube[6], 1.0, 1.2); // argument of pericenter
  double tp = prior_uniform(cube[7], 2002.1, 2002.5); // time of pericenter

  // coordinate system parameters
  double x0 = prior_uniform(cube[8], -5E-3, 5E-3);
  double y0 = prior_uniform(cube[9], -5E-3, 5E-3);
  double vx0 = prior_uniform(cube[10], -1E-3, 1E-3);
  double vy0 = prior_uniform(cube[11], -1E-3, 1E-3);
  double vz0 = prior_uniform(cube[12], 0., 50.);

  // set up orbit integration
  struct reb_simulation* r = reb_create_simulation();
  struct rebx_extras* rebx = rebx_init(r);
  r->G = gravitational_constant;
  r->integrator = REB_INTEGRATOR_IAS15;
  r->exact_finish_time = 1;
  r->N_active = 1; // only the black hole is massive

  // add black hole particle
  struct reb_particle bh_particle = {0};
  bh_particle.m = M0; // black hole mass
  bh_particle.hash = reb_hash("black_hole");
  reb_add(r, bh_particle);
  // create consistent pointer to particle
  struct reb_particle* bh = reb_get_particle_by_hash(r, reb_hash("black_hole"));

  // add star "test" particle
  r->t = 1990.; // orbit integration start time
  double n = sqrt((r->G * bh->m)/pow(a, 3)); // mean motion (P = 2pi/n)
  double M = n * (r->t - tp); // mean anomaly
  double f = reb_tools_M_to_f(e, M); // true anomaly
  struct reb_particle test_particle = reb_tools_orbit_to_particle(r->G, *bh, 0.,
                                        a, e, inc, Omega, omega, f);
  test_particle.hash = reb_hash("test_particle");
  reb_add(r, test_particle);
  struct reb_particle* p = reb_get_particle_by_hash(r, reb_hash("test_particle"));

  // enable post-Newtonian corrections
  struct rebx_effect* gr_params = rebx_add(rebx, "gr");
  double* gr_params_c = rebx_add_param(gr_params, "c", REBX_TYPE_DOUBLE);
  *gr_params_c = speed_of_light;
  int* gr_source = rebx_add_param(bh, "gr_source", REBX_TYPE_INT);
  *gr_source = 1;

  double value = 0;
  for (int i = 0; i < d->N_points; i++) {
    struct point obs = points[i];

    double t_obs = obs.t;
    reb_integrate(r, t_obs); // time of observation

    // account for the light propagation delay (Roemer effect)
    double t_emm = t_obs - (p->z/speed_of_light * (1. - p->vz/speed_of_light));
    reb_integrate(r, t_emm); // time of emission

    if (obs.type == position_type) {

      // convert to the coordinate system of the observations
      double x_pred = -p->y;
      double y_pred = p->x;

      // account for a possible drift of the astrometric reference frame
      x_pred += x0 + vx0 * (r->t - 2000.);
      y_pred += y0 + vy0 * (r->t - 2000.);

      value += -0.5*pow((x_pred - obs.x)/obs.x_err, 2)
               -0.5*log(2*M_PI * pow(obs.x_err, 2));
      value += -0.5*pow((y_pred - obs.y)/obs.y_err, 2)
               -0.5*log(2*M_PI * pow(obs.y_err, 2));
    }
    else if (obs.type == velocity_type) {

      // account for the relativistic Doppler effect
      double beta_costheta = p->vz/speed_of_light;
      double beta2 = (pow(p->vx, 2) + pow(p->vy, 2) + pow(p->vz, 2))/pow(speed_of_light, 2);
      double zD = (1. + beta_costheta)/sqrt(1. - beta2) - 1.;

      // account for the gravitational redshift
      double rs = 2.*sim->G*bh->m/pow(speed_of_light, 2);
      double zG = 1./sqrt(1. - rs/sqrt(pow(p->x, 2) + pow(p->y, 2) + pow(p->z, 2))) - 1.;

      // calculate the measured radial velocity
      double vz_pred = (zD + zG) * speed_of_light;

      // convert to observed units
      vz_pred *= velocity_conversion_factor;

      // account for a possible radial velocity offset
      vz_pred += vz0;

      value += -0.5*pow((vz_pred - obs.vz)/obs.vz_err, 2)
               -0.5*log(2*M_PI * pow(obs.vz_err, 2));
    }
  }

  *new_value = value;

  rebx_free(rebx);
  reb_free_simulation(r);
}

double prior_uniform(double x, double a, double b) {
  return a + (b - a)*x;
}

struct data* import_data(char* filename, int N_points) {
  struct data* d = calloc(1, sizeof(struct data));
  d->N_points = N_points;
  d->points = realloc(d->points, sizeof(struct point) * d->N_points);

  FILE* inf = fopen(filename, "r");

  for (int i = 0; i < d->N_points; i++) {
    struct point obs;
    fscanf(inf, "%le,%d,", &obs.t, &obs.type);
    if (obs.type == position_type) {
      double scale; // error scaling factor
      fscanf(inf, "%le,%le,%le,%le,,,%le\n",
        &obs.x, &obs.x_err, &obs.y, &obs.y_err, &scale);
      obs.x_err *= scale;
      obs.y_err *= scale;
    }
    else if (obs.type == velocity_type) {
      double scale;
      fscanf(inf, ",,,,%le,%le,%le\n", &obs.vz, &obs.vz_err, &scale);
      obs.vz_err *= scale;
    }
    else {
      abort();
    }
    d->points[i] = obs;
  }

  fclose(inf);

  return d;
}

void dumper(int *nsamples, int *n_live, int *n_pars, double **physLive, double **posterior,
            double **paramConstr, double *maxloglike, double *logZ, double *INSlogZ,
            double *logZerr, void *context) {
  // do nothing ...
}
