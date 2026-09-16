#include <iostream>
#include <cmath>
#include <cassert>
#include "adjoint.hpp"
#include "cfd.hpp"

template <typename T>
T test_function(const std::vector<T>& x) {
    T result(0.0);
    for (size_t i = 0; i < x.size(); i++) {
        result = result + exp(x[i]) * log(x[i] + T(2.0)) + sqrt(x[i]*x[i] + T(1.0));
    }
    return result;
}

std::vector<double> ad_gradient(const std::vector<double>& x) {
    g_tape<double>.reset();
    std::vector<Adjoint<double>> x_a;
    for (double v : x) x_a.push_back(Adjoint<double>(v));

    Adjoint<double> result = test_function(x_a);

    g_tape<double>.init_adjoints();
    g_tape<double>.seed_adjoint(result.idx, 1.0);
    g_tape<double>.propagate();

    std::vector<double> grad(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        grad[i] = g_tape<double>.get_adjoint(x_a[i].idx);
    }
    return grad;
}

std::vector<double> fd_gradient(std::vector<double>& x) {
    return cfd_gradient(
        [](std::vector<double>& x) { return test_function<double>(x); },
        x
    );
}

bool close(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) < tol;
}

int main(void) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};

    std::vector<double> gradAD = ad_gradient(x);
    std::vector<double> gradFD = fd_gradient(x);
    
    bool allMatch = true;
    for (size_t i = 0; i < x.size(); i++) {
        bool ok = close(gradAD[i], gradFD[i]);
        std::cout << "  x[" << i << "] = " << x[i]
                   << "   AD = " << gradAD[i]
                   << "   FD = " << gradFD[i]
                   << "   diff = " << std::fabs(gradAD[i] - gradFD[i])
                   << (ok ? "   [MATCH]" : "   [MISMATCH]") << "\n";
        allMatch = allMatch && ok;
    }

    std::cout << "-------------------------------------\n";
    std::cout << (allMatch ? "PASSED: all gradient components match.\n"
                            : "FAILED: mismatch detected — do not trust benchmark timings until resolved.\n");

    assert(allMatch && "AD gradient does not match FD gradient");

    return allMatch ? 0 : 1;
}