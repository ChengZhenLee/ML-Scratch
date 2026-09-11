#include <cassert>
#include <iostream>
#include <limits>

#include "ad.hpp"
#include "black_scholes.hpp"


using var = Var<double>;

auto& tape = g_tape<double>;
double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());

void test1_first_order_derivatives() {
    tape.reset();
    var Sv(100.0), Kv(100.0), rv(0.05), sigmav(0.2), tauv(1.0);
    var result = black_scholes_call(Sv, Kv, rv, sigmav, tauv);
    tape.init_adjoints();
    tape.seed_adjoint(result.idx, 1.0);
    tape.propagate();

    double adDelta = tape.get_adjoint(Sv.idx);        // dS/dV
    double adVega = tape.get_adjoint(sigmav.idx);     // dsigma/dV
    double adTheta = -tape.get_adjoint(tauv.idx);     // dt/dV = -dtau/dV
    double adRho = tape.get_adjoint(rv.idx);          // dr/dV

    double S = 100.0, K = 100.0, r = 0.05, sigma = 0.2, tau =1.0;
    double exactDelta = delta_call(S, K, r, sigma, tau);
    double exactVega = vega(S, K, r, sigma, tau);
    double exactTheta = theta_call(S, K, r, sigma, tau);
    double exactRho = rho_call(S, K, r, sigma, tau);

    assert(std::fabs(adDelta - exactDelta) < tolerance && "Delta is inaccurate");
    assert(std::fabs(adVega - exactVega) < tolerance && "Vega is inaccurate");
    assert(std::fabs(adTheta - exactTheta) < tolerance && "Theta is inaccurate");
    assert(std::fabs(adRho - exactRho) < tolerance && "Rho is inaccurate");

    std::cout << "Black-Scholes first order derivatives test passed\n";
}

void test2_second_order_derivatives() {
    tape.reset();
    var Sv(100.0), Kv(100.0), rv(0.05), sigmav(0.2), tauv(1.0);
    var delta = delta_call(Sv, Kv, rv, sigmav, tauv);
    tape.init_adjoints();
    tape.seed_adjoint(delta.idx, 1.0);
    tape.propagate();

    double adGamma = tape.get_adjoint(Sv.idx);
    double adVanna = tape.get_adjoint(sigmav.idx);
    double adCharm = -tape.get_adjoint(tauv.idx);

    double S = 100.0, K = 100.0, r = 0.05, sigma = 0.2, tau = 1.0;
    double exactGamma = gamma(S, K, r, sigma, tau);
    double exactVanna = vanna(S, K, r, sigma, tau);
    double exactCharm = charm(S, K, r, sigma, tau);

    assert(std::fabs(adGamma - exactGamma) < tolerance && "Gamma is inaccurate");
    assert(std::fabs(adVanna - exactVanna) < tolerance && "Vanna is inaccurate");
    assert(std::fabs(adCharm - exactCharm) < tolerance && "Charm is inaccurate");

    std::cout << "Black-Scholes second order derivatives test passed\n";
}

int main(void) {
    test1_first_order_derivatives();
    test2_second_order_derivatives();
    std::cout << "All Black-Scholes tests passed\n";

    return 0;
}