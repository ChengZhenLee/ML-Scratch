#include <iostream>
#include <cmath>
#include <cassert>
#include <limits>
#include <vector>

#include "Eigen/Dense"


struct MarkowitzReturn {
    Eigen::VectorXd w;
    double lambda1;
    double lambda2;

    MarkowitzReturn(Eigen::VectorXd w, double lambda1, double lambda2) :
        w(w), lambda1(lambda1), lambda2(lambda2) {}
};

std::vector<double> toVec(const Eigen::VectorXd& v) {
    return std::vector<double>(v.data(), v.data() + v.size());
}

// Overload the << operator for ostream to print out vectors
std::ostream& operator<<(std::ostream& os, const std::vector<double>& v) {
    os << "(";
    for (size_t j = 0; j < v.size(); j++) os << v[j] << (j + 1 < v.size() ? ", " : "");
    return os << ")";
}

MarkowitzReturn solve_markowitz(
    const Eigen::MatrixXd& Sigma, 
    const Eigen::VectorXd& mu, 
    double rTarget
) {
    Eigen::VectorXd w;

    Eigen::VectorXd ones = Eigen::VectorXd::Ones(mu.size());

    Eigen::MatrixXd Sigma_inverse = Sigma.inverse();
    double A = mu.transpose() * Sigma_inverse * mu;
    double B = mu.transpose() * Sigma_inverse * ones;
    double C = ones.transpose() * Sigma_inverse * ones;
    double D = A * C - B * B;

    double lambda1 = 2 * (C * rTarget - B) / D;
    double lambda2 = 2 * (A - B * rTarget) / D;
    w = ((C * rTarget - B) * Sigma_inverse * mu + (A - B * rTarget) * Sigma_inverse * ones) / D;

    return {w, lambda1, lambda2};
}


int main(void) {
    // Toy 3-asset portfolio

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
    Eigen::MatrixXd Sigma(3,3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Sigma(i, j) = rho(i, j) * sigma(i) * sigma(j);
        }
    }

    // Set the target returns
    double rTarget = 0.07;

    // Run the markowitz solver
    MarkowitzReturn result = solve_markowitz(Sigma, mu, rTarget);
    Eigen::VectorXd w = result.w;
    double lambda1 = result.lambda1;
    double lambda2 = result.lambda2;

    Eigen::VectorXd ones = Eigen::VectorXd::Ones(mu.size());
    double wSum = w.sum();
    double expectedR = w.transpose() * mu;
    Eigen::VectorXd gradient = 2 * Sigma * w - lambda1 * mu - lambda2 * ones;

    double tol = std::sqrt(std::numeric_limits<double>::epsilon());
    bool constraintsOk = std::fabs(wSum - 1.0) < tol && std::fabs(expectedR - rTarget) < tol;

    std::cout << "\n[Constraint check]\n";
    std::cout << "  Sum of weights:    " << wSum << "   (target: 1.0,  error: " << std::fabs(wSum - 1.0) << ")\n";
    std::cout << "  Expected return:   " << expectedR << "   (target: " << rTarget << ",  error: " << std::fabs(expectedR - rTarget) << ")\n";
    std::cout << "  Constraints approximately satisfied? " << (constraintsOk ? "YES" : "NO") << "\n";

    std::cout << "\n[Stationarity check]\n";
    std::cout << "  Lagrangian gradient: " << toVec(gradient) << "\n";
    std::cout << "  Gradient norm:  " << gradient.norm() << "   (should be small if converged)\n";

    std::cout << "\n[Result summary]\n";
    std::cout << "  Optimal weights: " << toVec(w) << "\n";
    std::cout << "  Expected return: " << expectedR << "\n";

    assert(std::fabs(wSum - 1.0) < tol && "weights should sum to 1");
    assert(std::fabs(expectedR - rTarget) < tol && "should hit target return");
    assert(gradient.norm() < tol && "Lagrangian condition should hold at optimal w");

    std::cout << "\nAll checks passed.\n";
    return 0;
}