#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "adjoint.hpp"
#include "tangent.hpp"


auto close = [](double a, double b, 
    double tol = std::sqrt(std::numeric_limits<double>::epsilon())) {
    return std::fabs(a - b) < tol;
};

// Tangent<Tangent<double>>
void test1_nested_tangent_over_tangent() {
    Tangent<double> x_seed(2.0);

    // Seed X^(2)
    x_seed.seed_tangent(1.0);

    Tangent<Tangent<double>> x(x_seed);

    // Seed X^(1)
    x.seed_tangent(Tangent<double>(1.0));

    Tangent<Tangent<double>> y = x * x * x;

    assert(close(y.value.value, 8.0));
    assert(close(y.tangent.value, 12.0));
    assert(close(y.tangent.value, y.value.tangent));
    assert(close(y.tangent.tangent, 12.0));

    std::cout << "test1 passed\n";
}

// Adjoint<Tangent<double>>
void test2_nested_tangent_over_adjoint() {
    g_tape<Tangent<double>>.reset();

    Tangent<double> x_seed(2.0);
    // Seed X^(2)
    x_seed.seed_tangent(1.0);

    // Wrap the types properly
    Adjoint<Tangent<double>> x(x_seed);

    Adjoint<Tangent<double>> y = x * x * x;

    g_tape<Tangent<double>>.init_adjoints();

    // Seed Y_(1)
    g_tape<Tangent<double>>.seed_adjoint(y.idx, Tangent<double>(1.0, 0.0));
    g_tape<Tangent<double>>.propagate();

    Tangent<double> result = g_tape<Tangent<double>>.get_adjoint(x.idx);
    // X_(1) is result.value, X_(1)^(2) is result.tangent

    assert(close(y.value.value, 8.0));       // the primal: x^3 = 8
    assert(close(y.value.tangent, 12.0));   // Check that the dy/dx = 12
    assert(close(y.value.tangent, result.value)); // Check that X_(1) and Y^(2) match
    assert(close(result.tangent, 12.0));    // second-order derivative X_(1)^(2): d2y/dx2 = 12, read via .value since tangent is itself a Var
    std::cout << "test2 passed\n";
}

// Tangent<Adjoint<double>>
void test3_nested_adjoint_over_tangent() {
    g_tape<double>.reset();

    // Wrap the types properly
    Adjoint<double> x_a(2.0);
    Tangent<Adjoint<double>> x_t(x_a);

    // Seed X^(1)
    x_t.seed_tangent(Adjoint<double>(1.0));

    // Primal function
    Tangent<Adjoint<double>> y_t = x_t * x_t * x_t;
    
    g_tape<double>.init_adjoints();
    // Seed Y^(1)_(2)
    g_tape<double>.seed_adjoint(y_t.tangent.idx, 1.0);
    g_tape<double>.propagate();

    // Second derivative will be in X^(2)
    double res = g_tape<double>.get_adjoint(x_t.value.idx);

    assert(close(y_t.value.value, 8.0));    // The primal value should be 8.0
    assert(close(y_t.tangent.value, 12.0)); // The first derivative is 12.0
    assert(close(y_t.tangent.value, g_tape<double>.get_adjoint(x_t.tangent.idx)));  // Check that X^(1)_(2) and Y^(1) match
    assert(close(res, 12.0));   // Check that the second derivative is 12.0

    std::cout << "test3 passed\n";
}


int main(void) {
    test1_nested_tangent_over_tangent();
    test2_nested_tangent_over_adjoint();
    test3_nested_adjoint_over_tangent();
}