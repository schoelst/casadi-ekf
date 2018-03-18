#include "ekf.hpp"

using namespace std;
using namespace casadi;

Function rungeKutta4(const string& name, SXDict& dae, const int M)
{
  // Dynamic step step Runge-Kutta 4 integrator
  SX dt = dae.at("dt");
  SX h = dt / M;  // step width
  vector<string> f_in = {"x", "p"};
  vector<string> f_out = {"ode", "quad"};
  Function f = Function("f", dae, f_in, f_out);

  // Number of differential states
  int nx = dae["x"].size1();

  // Number of controls
  int nu = dae["p"].size1();

  SX X0 = SX::sym("X0", nx);
  SX U_ = SX::sym("U", nu);
  SX X_ = X0;
  SX Q = 0;
  SXDict k1, k2, k3, k4;
  SXDict a1, a2, a3, a4;
  for (int i = 0; i < M; ++i)
  {
    a1["x"] = X_;
    a1["p"] = U_;
    f.call(a1, k1);
    a2["x"] = X_ + h / 2. * k1["ode"];
    a2["p"] = U_;
    f.call(a2, k2);
    a3["x"] = X_ + h / 2. * k2["ode"];
    a3["p"] = U_;
    f.call(a3, k3);
    a4["x"] = X_ + h * k3["ode"];
    a4["p"] = U_;
    f.call(a4, k4);
    X_ = X_ + h / 6. * (k1["ode"] + 2 * k2["ode"] + 2 * k3["ode"] + k4["ode"]);
    Q = Q + h / 6. * (k1["quad"] + 2 * k2["quad"] + 2 * k3["quad"] + k4["quad"]);
  }

  SXDict Dae = {{"x0", X0}, {"p", U_}, {"dt", dt}, {"xf", X_}, {"qf", Q}};
  vector<string> F_in = {"x0", "p", "dt"};
  vector<string> F_out = {"xf", "qf"};
  Function F = Function(name, Dae, F_in, F_out);
  return F;
}

CasadiEKF::CasadiEKF(EKFinitialConditions init):
  Nx(init.f.size1_in(0)),
  Nu(init.f.size1_in(1)),
  Ny(init.g.size1_out(0))
{
  // copy initial conditions
  x = init.x0;
  P = init.P0;

  // copy noise matrices
  Q = init.Q;
  R = init.R;

  // evaluate model functions
  SX x_var = SX::sym("x", Nx);
  SX u_var = SX::sym("u", Nu);
  vector<SX> ode;
  init.f.call({x_var, u_var}, ode);
  SX xDot = ode[0];
  vector<SX> meas;
  init.g.call({x_var, u_var}, meas);
  SX y = meas[0];

  // discretized model
  SX dt = SX::sym("dt");
  SXDict dae = {{"x", x_var}, {"p", u_var}, {"dt", dt}, {"ode", ode[0]}};
  F = rungeKutta4("F", dae, 1);
  SX xNext = F(SXDict({{"x", x_var}, {"p", u_var}, {"dt", dt}})).at("xf");

  // compute Jacobians
  SX A = SX::jacobian(xNext, x_var); // this should be wrt. F?
  SX C = SX::jacobian(y, x_var);
  jac_f = Function("jac_f", {{"x", x_var}, {"u", u_var}, {"dt", dt}, {"jac", A}}, {"x", "u"}, {"jac"});
  jac_g = Function("jac_g", {{"x", x_var}, {"u", u_var}, {"jac", C}}, {"x", "u"}, {"jac"});
}

void CasadiEKF::step(const vector<double>& y, const std::vector<double>& u, const double dt)
{
  // Prediction Step
  // TODO replace rungeKutta4 here with Ts
  x = F(DMDict({{"x", x}, {"u", u}, {"dt", dt}})).at("xf");
  A = jac_f(DMDict({{"x", x}, {"u", u}, {"dt", dt}})).at("jac");
  P = mtimes(A, mtimes(P, A.T())) + dt * R;

  // (Innovation) Update Step / correction step
  C = jac_g(DMDict({{"x", x}, {"u", u}})).at("jac");
  y_pred = g(DMDict({{"x", x}, {"u", u}})).at("y");
  P12 = mtimes(P, C.T());

  K = mtimes(P12, inv(mtimes(C, P12) + Q / dt));
  x += mtimes(K, (y - y_pred));
  P -= mtimes(mtimes(K, C), P);

}

void CasadiEKF::getState(std::vector<double>& x) const
{
  x = vector<double>(this->x);
}

void CasadiEKF::getCovariance(vector<vector<double>>& cov) const
{
  cov = vector<vector<double>>(Q);
}
