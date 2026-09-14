#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "adjoint.hpp"
#include "cfd.hpp"


auto close = [](double a, double b, 
    double tol = std::sqrt(std::numeric_limits<double>::epsilon())) {
    return std::fabs(a - b) < tol;
};

auto& tape = g_tape<double>;

void test1_single_multiply() {
    tape.reset();
    Adjoint a(2.0), b(3.0);
    Adjoint c = a * b;
    // tape.backward(c.idx);
    tape.init_adjoints();
    tape.seed_adjoint(c.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(a.idx), 3.0));  // dc/da = b
    assert(close(tape.get_adjoint(b.idx), 2.0));  // dc/db = a
    std::cout << "test1 passed\n";
}

void test2_reused_variable() {
    tape.reset();
    Adjoint a(2.0), b(3.0);
    Adjoint c = a * b + a;      // dc/da = b + 1, dc/db = a
    tape.init_adjoints();
    tape.seed_adjoint(c.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(a.idx), 4.0));
    assert(close(tape.get_adjoint(b.idx), 2.0));
    std::cout << "test2 passed\n";
}

void test3_chain_of_ops() {
    tape.reset();
    Adjoint x(2.0);
    Adjoint y = x * x * x;   // y = x^3, dy/dx = 3x^2 = 12
    tape.init_adjoints();
    tape.seed_adjoint(y.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(x.idx), 12.0));
    std::cout << "test3 passed\n";
}

void test4_division() {
    tape.reset();
    Adjoint a(6.0), b(2.0);
    Adjoint c = a / b;                 // c = 3
    tape.init_adjoints();
    tape.seed_adjoint(c.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(a.idx), 1.0/b.value));            // 0.5
    assert(close(tape.get_adjoint(b.idx), -a.value/(b.value*b.value))); // -1.5
    std::cout << "test4 passed\n";
}

void test5_diamond_graph() {
    tape.reset();
    Adjoint a(2.0);
    Adjoint p = a * a;         // p = a^2
    Adjoint q = a + a;         // q = 2a
    Adjoint r = p * q;         // r = a^2 * 2a = 2a^3, dr/da = 6a^2 = 24
    tape.init_adjoints();
    tape.seed_adjoint(r.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(a.idx), 24.0));
    std::cout << "test5 passed\n";
}

// A dummy function
template <typename T>
T f(std::vector<T>& v) {
    T x = v[0];
    T y = v[1];
    return x * y + x / y;
}

void test6_finite_difference_crosscheck() {
    double x = 1.7, y = 0.9;
    Adjoint vx = Adjoint(x), vy = Adjoint(y);
    std::vector<Adjoint<double>> vInputs = {vx, vy};

    auto out = f(vInputs);
    tape.init_adjoints();
    tape.seed_adjoint(out.idx, 1.0);
    tape.propagate();
    double ad_dx = tape.get_adjoint(vx.idx);
    double ad_dy = tape.get_adjoint(vy.idx);

    std::vector<double> cfdInputs = {x, y};
    std::vector<double> gradients = cfd_gradient(f<double>, cfdInputs);
    double fd_dx = gradients[0];
    double fd_dy = gradients[1];

    assert(close(ad_dx, fd_dx, 1e-6));
    assert(close(ad_dy, fd_dy, 1e-6));
    std::cout << "test6 passed (AD vs finite difference)\n";
}

void test7_mixed_var_constant() {
    tape.reset();
    Adjoint<double> x(3.0);
    Adjoint<double> y = x * 2.0 + 1.0;      // y = 2x + 1, dy/dx = 2
    tape.init_adjoints();
    tape.seed_adjoint(y.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(x.idx), 2.0));
    assert(close(y.value, 7.0));

    tape.reset();
    Adjoint<double> a(4.0);
    Adjoint<double> b = 10.0 / a;           // b = 10/a, db/da = -10/a^2 = -0.625
    tape.init_adjoints();
    tape.seed_adjoint(b.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(a.idx), -10.0/(4.0*4.0)));
    std::cout << "test7 passed\n";
}

void test8_comparisons_dont_touch_tape() {
    tape.reset();

    Adjoint<double> x(3.0), y(5.0);
    size_t nodes_before = tape.nodes.size();

    bool r1 = x < y;
    bool r2 = y > x;
    bool r3 = x < 10.0;

    assert(r1 == true);
    assert(r2 == true);
    assert(r3 == true);

    size_t nodes_after = tape.nodes.size();
    assert(nodes_before == nodes_after);
    std::cout << "test8 passed\n";
}

int main(void) {
    test1_single_multiply();
    test2_reused_variable();
    test3_chain_of_ops();
    test4_division();
    test5_diamond_graph();
    test6_finite_difference_crosscheck();
    test7_mixed_var_constant();
    test8_comparisons_dont_touch_tape();
    std::cout << "all tests passed\n";
}