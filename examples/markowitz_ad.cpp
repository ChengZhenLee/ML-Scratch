#include <iostream>
#include <cmath>
#include <cassert>
#include <limits>

#include "adjoint.hpp"
#include "Eigen/Dense"


double stepSize = 0.0005;
int maxIter = 50000;
double lambdaPenalty = 500.0;

template <typename T>
T penalized_objective(
    const std::vector<T> w, 
    Eigen::MatrixXd Sigma, 
    Eigen::VectorXd mu, 
    double rTarget
) 
{
    // Calculate the portfolio variance
    T variance = T(0.0);
    for (int i = 0 ; i < w.size(); i++) {
        for (int j = 0; j < w.size(); j++) {
            variance = variance + Sigma(i, j) * w[i] * w[j];
        }
    }

    // Calculate the portfolio expected returns
    T returns = T(0.0);
    for (int i = 0; i < w.size(); i++) {
        returns = returns + w[i] * mu(i);
    }

    // Calculate the portfolio sum
    T sum = T(0.0);
    for (int i = 0; i < w.size(); i++) {
        sum = sum + w[i];
    }

    return variance
        + lambdaPenalty * pow(returns - rTarget, 2)
        + lambdaPenalty * pow(sum - 1.0, 2);
}


int main(void) {
    // Toy 3-asset portfolio

    // Initial weights
    Eigen::VectorXd w(3);
    w << 1.0/3.0, 1.0/3.0, 1.0/3.0;

    // Expected returns
    Eigen::VectorXd mu(3);
    mu << 0.03, 0.08, 0.12;

    // Standard deviations
    Eigen::VectorXd sigma(3);
    sigma << 0.05, 0.20, 0.30;

    // Correlation matrix
    Eigen::MatrixXd rho(3, 3);
    rho << 1.0, 0.1, 0.1,
           0.1, 1.0, 0.5,
           0.1, 0.5, 1.0;

    // Covariance_a_b = Correlation_a_b * standard_deviation_a * standard_deviation_b
    Eigen::MatrixXd Sigma(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Sigma(i, j) = rho(i, j) * sigma(i) * sigma(j);
        }
    }

    // Set the target returns
    double rTarget = 0.07;

    Eigen::VectorXd grad(3);
    for (int i = 0; i < maxIter; i++) {
        g_tape<double>.reset();
        std::vector<Adjoint<double>> w_a;
        for (int i = 0; i < 3; i++) {
            w_a.push_back(Adjoint<double>(w(i)));
        }

        Adjoint<double> penalizedObjective = penalized_objective(w_a, Sigma, mu, rTarget);
        g_tape<double>.init_adjoints();
        g_tape<double>.seed_adjoint(penalizedObjective.idx, 1.0);
        g_tape<double>.propagate();

        // Gradient descent
        for (int i = 0; i < 3; i++) {
            grad(i) = g_tape<double>.get_adjoint(w_a[i].idx);
        }
        w = w - stepSize * grad;
    }

    double wSum = w.sum();
    double expectedR = w.transpose() * mu;
    std::cout << "Optimal weights:\n" << w << "\n\n";
    std::cout << "Sum of weights:    " << wSum << "  (should be 1.0)\n";
    std::cout << "Expected return:   " << expectedR << "  (target was " << rTarget << ")\n";
    std::cout << "Lagrangian gradient norm: " << grad.norm() << "  (should be ~0)\n";

    double tol = 1e-3;
    assert(std::fabs(wSum - 1.0) < tol && "weights should sum to 1");
    assert(std::fabs(expectedR - rTarget) < tol && "should hit target return");
    assert(grad.norm() < tol && "Lagrangian condition should hold at optimal w");

    std::cout << "\nAll checks passed.\n";
    return 0;
}