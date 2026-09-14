#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "adjoint.hpp"
#include "tangent.hpp"
#include "cfd.hpp"


auto close = [](double a, double b, 
    double tol = std::sqrt(std::numeric_limits<double>::epsilon())) {
    return std::fabs(a - b) < tol;
};

void test1_tangent_multiply() {
    Tangent<double> x{2.0};
    Tangent<double> y{3.0};

    x.seed_tangent(1.0);

    Tangent<double> z = x * y;      // z = x*y, dz/dx = y = 3

    assert(close(z.value, 6.0));
    assert(close(z.tangent, 3.0));
    std::cout << "test1 passed\n";
}

void test2_tangent_chain() {
    Tangent<double> x{2.0};

    x.seed_tangent(1.0);

    Tangent<double> y = x * x * x;      // y = x^3, dy/dx = 3x^2 = 12

    assert(close(y.value, 8.0));
    assert(close(y.tangent, 12.0));
    std::cout << "test2 passed\n";
}

void test3_dual_transcendental() {
    Tangent<double> x{2.0};

    x.seed_tangent(1.0);

    Tangent<double> y = exp(x) * log(x);

    double expected_value = std::exp(2.0) * std::log(2.0);
    double expected_tangent = std::exp(2.0)*std::log(2.0) + std::exp(2.0)*(1.0/2.0);  // product rule
    assert(close(y.value, expected_value));
    assert(close(y.tangent, expected_tangent));
    std::cout << "test3 passed\n";
}

void test4_forward_vs_reverse_agreement() {
    // reverse mode
    g_tape<double>.reset();
    Adjoint<double> vx(2.0), vy(3.0);
    Adjoint<double> vz = vx * vy + vx / vy;

    g_tape<double>.init_adjoints();
    g_tape<double>.seed_adjoint(vz.idx, 1.0);
    g_tape<double>.propagate();

    double rev_dz_dx = g_tape<double>.get_adjoint(vx.idx);
    double rev_dz_dy = g_tape<double>.get_adjoint(vy.idx);

    // forward mode, one pass per input
    Tangent<double> dx1{2.0}, dy1{3.0};
    dx1.seed_tangent(1.0);
    Tangent<double> dz1 = dx1 * dy1 + dx1 / dy1;
    double fwd_dz_dx = dz1.tangent;

    Tangent<double> dx2{2.0}, dy2{3.0};
    dy2.seed_tangent(1.0);
    Tangent<double> dz2 = dx2 * dy2 + dx2 / dy2;
    double fwd_dz_dy = dz2.tangent;

    assert(close(rev_dz_dx, fwd_dz_dx));
    assert(close(rev_dz_dy, fwd_dz_dy));
    std::cout << "test4 passed\n";
}

// A dummy function
template <typename T>
T f(std::vector<T>& v) {
    T x = v[0];
    T y = v[1];
    return x * y + x / y;
}

void test5_dual_vs_cfd() {
    std::vector<double> inputs = {1.7, 0.9};

    // forward mode: one pass per input, seeding one tangent = 1 at a time
    Tangent<double> dx1{inputs[0]}, dy1{inputs[1]};
    dx1.seed_tangent(1.0);
    Tangent<double> out_dx = dx1 * dy1 + dx1 / dy1;
    double fwd_ddx = out_dx.tangent;

    Tangent<double> dx2{inputs[0]}, dy2{inputs[1]};
    dy2.seed_tangent(1.0);
    Tangent<double> out_dy = dx2 * dy2 + dx2 / dy2;
    double fwd_ddy = out_dy.tangent;

    std::vector<double> cfd_inputs = inputs;
    std::vector<double> grads = cfd_gradient(f<double>, cfd_inputs);

    assert(close(fwd_ddx, grads[0], 1e-6));
    assert(close(fwd_ddy, grads[1], 1e-6));
    std::cout << "test5 passed\n";
}


int main(void) {
    test1_tangent_multiply();
    test2_tangent_chain();
    test3_dual_transcendental();
    test4_forward_vs_reverse_agreement();
    test5_dual_vs_cfd();
}