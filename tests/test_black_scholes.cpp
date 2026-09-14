#include <cassert>
#include <iostream>
#include <limits>

#include "adjoint.hpp"
#include "tangent.hpp"
#include "black_scholes.hpp"


double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());

auto close = [] (double a, double b) {
    return std::fabs(a - b) < tolerance;
};


void test1_first_order_derivatives() {
    g_tape<double>.reset();
    Adjoint<double> Sv(100.0), Kv(100.0), rv(0.05), sigmav(0.2), tauv(1.0);
    Adjoint<double> result = black_scholes_call(Sv, Kv, rv, sigmav, tauv);
    g_tape<double>.init_adjoints();
    g_tape<double>.seed_adjoint(result.idx, 1.0);
    g_tape<double>.propagate();

    double adDelta = g_tape<double>.get_adjoint(Sv.idx);        // dS/dV
    double adVega = g_tape<double>.get_adjoint(sigmav.idx);     // dsigma/dV
    double adTheta = -g_tape<double>.get_adjoint(tauv.idx);     // dt/dV = -dtau/dV
    double adRho = g_tape<double>.get_adjoint(rv.idx);          // dr/dV

    double S = 100.0, K = 100.0, r = 0.05, sigma = 0.2, tau =1.0;
    double exactDelta = delta_call(S, K, r, sigma, tau);
    double exactVega = vega(S, K, r, sigma, tau);
    double exactTheta = theta_call(S, K, r, sigma, tau);
    double exactRho = rho_call(S, K, r, sigma, tau);

    assert(close(adDelta, exactDelta) && "Delta is inaccurate");
    assert(close(adVega, exactVega) && "Vega is inaccurate");
    assert(close(adTheta, exactTheta) && "Theta is inaccurate");
    assert(close(adRho, exactRho) && "Rho is inaccurate");

    std::cout << "Black-Scholes first order derivatives test passed\n";
}

void test2_second_order_derivatives() {
    g_tape<double>.reset();
    Adjoint<double> Sv(100.0), Kv(100.0), rv(0.05), sigmav(0.2), tauv(1.0);
    Adjoint<double> delta = delta_call(Sv, Kv, rv, sigmav, tauv);
    g_tape<double>.init_adjoints();
    g_tape<double>.seed_adjoint(delta.idx, 1.0);
    g_tape<double>.propagate();

    double adGamma = g_tape<double>.get_adjoint(Sv.idx);
    double adVanna = g_tape<double>.get_adjoint(sigmav.idx);
    double adCharm = -g_tape<double>.get_adjoint(tauv.idx);

    double S = 100.0, K = 100.0, r = 0.05, sigma = 0.2, tau = 1.0;
    double exactGamma = gamma(S, K, r, sigma, tau);
    double exactVanna = vanna(S, K, r, sigma, tau);
    double exactCharm = charm(S, K, r, sigma, tau);

    assert(close(adGamma, exactGamma) && "Gamma is inaccurate");
    assert(close(adVanna, exactVanna) && "Vanna is inaccurate");
    assert(close(adCharm, exactCharm) && "Charm is inaccurate");

    std::cout << "Black-Scholes second order derivatives test passed\n";
}

// Using Tangent over Adjoint mode
void test3_nested_ad_gamma() {
    g_tape<Tangent<double>>.reset();

    double S = 100.0, K = 100.0, r = 0.05, sigma = 0.2, tau = 1.0;
    Tangent<double> S_t(S), K_t(K), r_t(r), sigma_t(sigma), tau_t(tau);

    // Seed X^(2)
    S_t.seed_tangent(1.0);
    K_t.seed_tangent(0.0); r_t.seed_tangent(0.0); sigma_t.seed_tangent(0.0); tau_t.seed_tangent(0.0);

    // Nest the types
    Adjoint<Tangent<double>> S_t_a(S_t), K_t_a(K_t), r_t_a(r_t), sigma_t_a(sigma_t), tau_t_a(tau_t);

    // Run the primal function
    Adjoint<Tangent<double>> result = black_scholes_call(S_t_a, K_t_a, r_t_a, sigma_t_a, tau_t_a);
    g_tape<Tangent<double>>.init_adjoints();

    // Seed Y_(1)
    g_tape<Tangent<double>>.seed_adjoint(result.idx, Tangent<double>(1.0, 0.0));
    g_tape<Tangent<double>>.propagate();

    Tangent<double> delta = g_tape<Tangent<double>>.get_adjoint(S_t_a.idx);
    double adDelta = delta.value;
    double adGamma = delta.tangent;

    double exactDelta = delta_call(S, K, r, sigma, tau);
    double exactGamma = gamma(S, K, r, sigma, tau);

    assert(close(adDelta, exactDelta) && "Nested AD Delta (first-order) is inaccurate");
    assert(close(adGamma, exactGamma) && "Nested AD Gamma is inaccurate");

    std::cout << "Black-Scholes Gamma calculation via nested AD passed\n";
}

// Using Tangent over Adjoint mode
void test4_nested_ad_vanna() {
    g_tape<Tangent<double>>.reset();

    double S = 100.0, K = 100.0, r = 0.05, sigma = 0.2, tau = 1.0;
    Tangent<double> S_t(S), K_t(K), r_t(r), sigma_t(sigma), tau_t(tau);

    // Seed X^(2)
    sigma_t.seed_tangent(1.0);
    S_t.seed_tangent(0.0); K_t.seed_tangent(0.0); r_t.seed_tangent(0.0); tau_t.seed_tangent(0.0);

    // Nest the types
    Adjoint<Tangent<double>> S_t_a(S_t), K_t_a(K_t), r_t_a(r_t), sigma_t_a(sigma_t), tau_t_a(tau_t);

    // Run the primal function
    Adjoint<Tangent<double>> result = black_scholes_call(S_t_a, K_t_a, r_t_a, sigma_t_a, tau_t_a);
    g_tape<Tangent<double>>.init_adjoints();

    // Seed Y_(1)
    g_tape<Tangent<double>>.seed_adjoint(result.idx, Tangent<double>(1.0, 0.0));
    g_tape<Tangent<double>>.propagate();

    Tangent<double> delta = g_tape<Tangent<double>>.get_adjoint(S_t_a.idx);
    double adDelta = delta.value;
    double adVanna = delta.tangent;

    double exactDelta = delta_call(S, K, r, sigma, tau);
    double exactVanna = vanna(S, K, r, sigma, tau);

    assert(close(adDelta, exactDelta) && "Nested AD Delta (first-order) is inaccurate");
    assert(close(adVanna, exactVanna) && "Nested AD Vanna is inaccurate");

    std::cout << "Black-Scholes Vanna calculation via nested AD passed\n";
}

// Using Tangent over Adjoint mode
void test5_nested_ad_charm() {
    g_tape<Tangent<double>>.reset();

    double S = 100.0, K = 100.0, r = 0.05, sigma = 0.2, tau = 1.0;
    Tangent<double> S_t(S), K_t(K), r_t(r), sigma_t(sigma), tau_t(tau);

    // Seed X^(2)
    tau_t.seed_tangent(1.0);
    S_t.seed_tangent(0.0); K_t.seed_tangent(0.0); sigma_t.seed_tangent(0.0); r_t.seed_tangent(0.0);

    // Nest the types
    Adjoint<Tangent<double>> S_t_a(S_t), K_t_a(K_t), r_t_a(r_t), sigma_t_a(sigma_t), tau_t_a(tau_t);

    // Run the primal function
    Adjoint<Tangent<double>> result = black_scholes_call(S_t_a, K_t_a, r_t_a, sigma_t_a, tau_t_a);
    g_tape<Tangent<double>>.init_adjoints();

    // Seed Y_(1)
    g_tape<Tangent<double>>.seed_adjoint(result.idx, Tangent<double>(1.0, 0.0));
    g_tape<Tangent<double>>.propagate();

    Tangent<double> delta = g_tape<Tangent<double>>.get_adjoint(S_t_a.idx);
    double adDelta = delta.value;

    // Note that the Charm is -dDelta/dTau
    double adCharm = -delta.tangent;

    double exactDelta = delta_call(S, K, r, sigma, tau);
    double exactCharm = charm(S, K, r, sigma, tau);

    assert(close(adDelta, exactDelta) && "Nested AD Delta (first-order) is inaccurate");
    assert(close(adCharm, exactCharm) && "Nested AD Charm is inaccurate");

    std::cout << "Black-Scholes Charm calculation via nested AD passed\n";
}

int main(void) {
    test1_first_order_derivatives();
    test2_second_order_derivatives();
    test3_nested_ad_gamma();
    test4_nested_ad_vanna();
    test5_nested_ad_charm();

    return 0;
}