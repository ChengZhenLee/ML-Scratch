#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "ad.hpp"


auto close = [](double a, double b, 
    double tol = std::sqrt(std::numeric_limits<double>::epsilon())) {
    return std::fabs(a - b) < tol;
};

auto& tape = g_tape<double>;

void test1_single_multiply() {
    tape.reset();
    Var a(2.0), b(3.0);
    Var c = a * b;
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
    Var a(2.0), b(3.0);
    Var c = a * b + a;      // dc/da = b + 1, dc/db = a
    tape.init_adjoints();
    tape.seed_adjoint(c.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(a.idx), 4.0));
    assert(close(tape.get_adjoint(b.idx), 2.0));
    std::cout << "test2 passed\n";
}

void test3_chain_of_ops() {
    tape.reset();
    Var x(2.0);
    Var y = x * x * x;   // y = x^3, dy/dx = 3x^2 = 12
    tape.init_adjoints();
    tape.seed_adjoint(y.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(x.idx), 12.0));
    std::cout << "test3 passed\n";
}

void test4_division() {
    tape.reset();
    Var a(6.0), b(2.0);
    Var c = a / b;                 // c = 3
    tape.init_adjoints();
    tape.seed_adjoint(c.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(a.idx), 1.0/b.value));            // 0.5
    assert(close(tape.get_adjoint(b.idx), -a.value/(b.value*b.value))); // -1.5
    std::cout << "test4 passed\n";
}

void test5_diamond_graph() {
    tape.reset();
    Var a(2.0);
    Var p = a * a;         // p = a^2
    Var q = a + a;         // q = 2a
    Var r = p * q;         // r = a^2 * 2a = 2a^3, dr/da = 6a^2 = 24
    tape.init_adjoints();
    tape.seed_adjoint(r.idx, 1.0);
    tape.propagate();
    assert(close(tape.get_adjoint(a.idx), 24.0));
    std::cout << "test5 passed\n";
}

void test6_finite_difference_crosscheck() {
    auto f = [](double x, double y) {
        tape.reset();
        Var vx(x), vy(y);
        Var out = vx * vy + vx / vy;
        return std::make_tuple(out, vx, vy);
    };

    double x = 1.7, y = 0.9, h = 1e-6;

    auto [out0, vx0, vy0] = f(x, y);
    tape.init_adjoints();
    tape.seed_adjoint(out0.idx, 1.0);
    tape.propagate();
    double ad_dx = tape.get_adjoint(vx0.idx);
    double ad_dy = tape.get_adjoint(vy0.idx);

    auto [out_xp, dummy1, dummy2] = f(x + h, y);
    auto [out_xm, dummy3, dummy4] = f(x - h, y);
    double fd_dx = (out_xp.value - out_xm.value) / (2 * h);

    auto [out_yp, dummy5, dummy6] = f(x, y + h);
    auto [out_ym, dummy7, dummy8] = f(x, y - h);
    double fd_dy = (out_yp.value - out_ym.value) / (2 * h);

    assert(close(ad_dx, fd_dx, 1e-6));
    assert(close(ad_dy, fd_dy, 1e-6));
    std::cout << "test6 passed (AD vs finite difference)\n";
}


int main(void) {
    test1_single_multiply();
    test2_reused_variable();
    test3_chain_of_ops();
    test4_division();
    test5_diamond_graph();
    test6_finite_difference_crosscheck();
    std::cout << "all tests passed\n";
}