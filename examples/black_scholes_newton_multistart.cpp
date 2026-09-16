#include <vector>
#include <random>
#include <iostream>

#include "adjoint.hpp"
#include "tangent.hpp"
#include "black_scholes.hpp"
#include "Eigen/Dense"


const int simIter = 500;
const double lambdaPenalty = 50.0;
std::mt19937_64 engine;
std::normal_distribution<double> dist(0, 1);

// Per-underlying market parameters shared by every option in the book.
struct MarketParams {
    double S0, mu, sigmaStock, horizon;
};

// Fixed parameters of the options
struct OptionsBook {
    std::vector<double> K, r, sigma, tau;
    std::vector<int> isCall;
};

template <typename T>
T portfolio_value(const std::vector<T>& w, double S, const OptionsBook& book) {
    T value = T(0.0);
    for (size_t i = 0; i < w.size(); i++) {
        if (book.isCall[i]) {
            value = value + w[i] * T(black_scholes_call(S, book.K[i], book.r[i], book.sigma[i], book.tau[i]));
        } else {
            value = value + w[i] * T(black_scholes_put(S, book.K[i], book.r[i], book.sigma[i], book.tau[i]));
        }
    }

    return value;
}

std::vector<double> generate_paths(const MarketParams& market, int numPaths) {
    std::vector<double> paths(numPaths);
    for (int i = 0; i < numPaths; i++) {
        double Z = dist(engine);
        paths[i] = market.S0 * exp((market.mu - market.sigmaStock * market.sigmaStock / 2) * market.horizon
                                    + market.sigmaStock * sqrt(market.horizon) * Z);
    }
    return paths;
}

template <typename T>
struct RiskResult {
    T mean;
    T variance;
};

template <typename T>
RiskResult<T> portfolio_risk_return(const std::vector<T>& w, const std::vector<double>& paths, const OptionsBook& book) {
    T sum = T(0.0);
    T sumSqr = T(0.0);

    for (double S_T : paths) {
        T price = portfolio_value(w, S_T, book);
        sum = sum + price;
        sumSqr = sumSqr + price * price;
    }

    // Var = E(X^2) - (E(X))^2
    T n = T(double(paths.size()));
    T mean = sum / n;
    T meanSqr = sumSqr / n;
    T variance = meanSqr - mean * mean;

    return {mean, variance};
}

template <typename T>
T penalized_objective(
    const std::vector<T>& w, const std::vector<double>& paths, const OptionsBook& book,
    double valueTarget
) {
    auto result = portfolio_risk_return(w, paths, book);

    T weightSum = T(0.0);
    for (const auto& wi : w) weightSum = weightSum + wi;

    T returnViolation = result.mean - T(valueTarget);
    T budgetViolation = weightSum - T(1.0);

    return result.variance
        + T(lambdaPenalty) * returnViolation * returnViolation
        + T(lambdaPenalty) * budgetViolation * budgetViolation;
}

Eigen::MatrixXd compute_hessian(
    const std::vector<double>& w,
    const std::vector<double>& paths,
    const OptionsBook& book,
    double valueTarget
) {
    int n = w.size();
    Eigen::MatrixXd H(n, n);

    for (int col = 0; col < n; col++) {
        g_tape<Tangent<double>>.reset();

        std::vector<Adjoint<Tangent<double>>> w_a;
        for (int i = 0; i < n; i++) {
            Tangent<double> w_t(w[i]);
            w_t.seed_tangent(i == col ? 1.0 : 0.0);
            w_a.push_back(Adjoint<Tangent<double>>(w_t));
        }

        Adjoint<Tangent<double>> obj = penalized_objective(w_a, paths, book, valueTarget);

        g_tape<Tangent<double>>.init_adjoints();
        g_tape<Tangent<double>>.seed_adjoint(obj.idx, 1.0);
        g_tape<Tangent<double>>.propagate();

        for (int row = col; row < n; row++) {
            Tangent<double> adj = g_tape<Tangent<double>>.get_adjoint(w_a[row].idx);
            H(row, col) = adj.tangent;
            if (row != col) H(col, row) = adj.tangent;
        }
    }
    return H;
}


Eigen::VectorXd compute_gradient(
    const std::vector<double>& w,
    const std::vector<double>& paths,
    const OptionsBook& book,
    double valueTarget
) {
    g_tape<double>.reset();

    std::vector<Adjoint<double>> w_a;
    for (double wi : w) w_a.push_back(Adjoint<double>(wi));

    Adjoint<double> obj = penalized_objective(w_a, paths, book, valueTarget);
    g_tape<double>.init_adjoints();
    g_tape<double>.seed_adjoint(obj.idx, 1.0);
    g_tape<double>.propagate();

    Eigen::VectorXd grad(w.size());
    for (size_t j = 0; j < w.size(); j++) grad(j) = g_tape<double>.get_adjoint(w_a[j].idx);
    return grad;
}

double squared_norm(const Eigen::VectorXd& v) {
    return v.squaredNorm();
}

double sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

// Overload the << operator for ostream to print out vectors
std::ostream& operator<<(std::ostream& os, const std::vector<double>& v) {
    os << "(";
    for (size_t j = 0; j < v.size(); j++) os << v[j] << (j + 1 < v.size() ? ", " : "");
    return os << ")";
}


int main(void) {
    // Use same underlying stock for all options
    MarketParams market{100.0, 0.08, 0.22, 1.0 / 52.0};   // S0, mu, sigmaStock, horizon (one week ahead)
    double valueTarget = 120.0;

    OptionsBook book{
        {95.0,  105.0, 95.0,  105.0},   // K
        {0.05,  0.05,  0.05,  0.05},    // r
        {0.20,  0.20,  0.25,  0.25},    // sigma
        {0.5,   0.5,   1.0,   1.0},     // tau
        {1,     1,     0,     0},       // isCall
    };

    // Use several different starting points
    std::vector<std::vector<double>> startingPoints = {
        {0.25, 0.25, 0.25, 0.25},   // equal-weighted (what you've used so far)
        {0.70, 0.10, 0.10, 0.10},   // concentrated in position 1
        {0.10, 0.10, 0.10, 0.70},   // concentrated in position 4
    };

    std::vector<std::vector<double>> finalWeights(startingPoints.size());

    std::vector<Eigen::VectorXd> grads;
    std::vector<double> means, variances;

    std::vector<double> trainPaths = generate_paths(market, simIter);

    for (int i = 0; i < startingPoints.size(); i++) {
        std::vector<double> w = startingPoints[i];
        std::cout << "Started at " << w << "\n";

        Eigen::MatrixXd H = compute_hessian(w, trainPaths, book, valueTarget);
        Eigen::VectorXd grad = compute_gradient(w, trainPaths, book, valueTarget);

        Eigen::VectorXd w_v(w.size());
        for (size_t i = 0; i < w.size(); i++) {
            w_v(i) = w[i];
        }

        // LDLT decomposition to compute H^-1 * grad
        Eigen::VectorXd delta = H.ldlt().solve(grad);

        // Single Newton Step
        Eigen::VectorXd w_star = w_v - delta;

        std::vector<double> w_final;
        for (int j = 0; j < w_star.size(); j++) {
            w_final.push_back(w_star(j));
        }
        finalWeights[i] = w_final;

        std::cout << "  -> converged to " << w_final << "\n";

        auto [finalMean, finalVariance] = portfolio_risk_return<double>(w_final, trainPaths, book);
        Eigen::VectorXd finalGrad = compute_gradient(w_final, trainPaths, book, valueTarget);

        grads.push_back(finalGrad);
        means.push_back(finalMean);
        variances.push_back(finalVariance);
    }

    double gradTol = 1e-4;
    double relTol = 0.02;
    for (size_t k = 0; k < startingPoints.size(); k++) {
        double gradNorm = std::sqrt(squared_norm(grads[k]));
        bool wellConverged = gradNorm < gradTol;

        std::cout << "\nStart " << k << ": " << startingPoints[k] << "\n";
        std::cout << "  Final weights: " << finalWeights[k] << "\n";
        std::cout << "  Final variance:      " << variances[k] << "\n";
        std::cout << "  Final mean:          " << means[k] << "\n";
        std::cout << "  Final gradient norm: " << gradNorm
                << (wellConverged ? "  [CONVERGED]" : "  [NOT FULLY CONVERGED -- interpret comparison with caution]") << "\n";
    }

    // Consistency check, relative to Start 0, gated by whether both runs actually converged
    bool allConsistent = true;
    bool allConverged = true;
    std::cout << "\n[Consistency check: comparing all runs to Start 0]\n";

    for (size_t k = 0; k < variances.size(); k++) {
        double gk = std::sqrt(squared_norm(grads[k]));
        if (gk >= gradTol) allConverged = false;
    }

    for (size_t k = 1; k < variances.size(); k++) {
        double relDiff = std::fabs(variances[k] - variances[0]) / std::fabs(variances[0]);
        bool consistent = relDiff < relTol;

        std::cout << "  Start " << k << " vs Start 0: relative variance diff = " << relDiff
                << "  (abs: " << std::fabs(variances[k]-variances[0]) << ")"
                << (consistent ? "  [CONSISTENT]" : "  [DIFFERENT]") << "\n";

        allConsistent = allConsistent && consistent;
    }
}