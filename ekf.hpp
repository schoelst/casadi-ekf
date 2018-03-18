#include <vector>
#include <casadi/casadi.hpp>

struct EKFinitialConditions{
  casadi::Function f; // ode of your system
  casadi::Function g; // measurement function

  std::vector<double> x0; // assumed initial state of your system
  std::vector<std::vector<double>> P0; // initial estimate of covariance matrix

  std::vector<std::vector<double>> Q; // covariance of the measurement noise (assumed to be Gaussian and zero mean)
  std::vector<std::vector<double>> R; // covariance of the state noise (assumed to be Gaussian and zero mean)
};

/**
 * @brief: Extended Kalman Filter that uses CasADi to compute derivatives and linear algebra.
 *
 * Assuming the following model structure:
 *
 * xDot = f(x, u) + r
 * y = g(x, u) + q
 *
 * where x is the system's state, xDot is it's derivative, u is the controls applied to the system, y is a vector of all measurements stacked, r is a disturbance of the state equation (state noise) and is assumed to be zero mean and normally distributed with covariance R, q models the measurement noise also as zero mean and normally distributed noise with covariance Q.
 */
class CasadiEKF {

  CasadiEKF(EKFinitialConditions init);
  ~CasadiEKF();

  void step(const std::vector<double>& y, const std::vector<double>& u, const double dt);
  void getState(std::vector<double>& x) const;
  void getCovariance(std::vector<std::vector<double>>& cov) const;



private:
  CasadiEKF(); // default construction impossible

  const size_t Nx; // number of states
  const size_t Nu; // number of controls
  const size_t Ny; // number of measurements

  casadi::DM x; // current estimate of the state vector
  casadi::DM P; // current estimate of the covariance matrix

  casadi::DM R;
  casadi::DM Q;

  casadi::Function g;
  casadi::Function jac_f; // jacobian of the system's ode with respect to x
  casadi::Function jac_g; // jacobian of measurement function

  casadi::Function F; // discrete time model of you the system

  // temporary variables
  casadi::DM A, C, P12, K, y_pred;
};
