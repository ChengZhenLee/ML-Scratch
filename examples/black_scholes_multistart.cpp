#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "adjoint.hpp"
#include "black_scholes.hpp"


int maxIter = 8000;
int simIter = 500;
double stepSize = 0.0005;
double c = 1e-4;
double gradTol = 1e-4;
std::mt19937_64 engine;
std::normal_distribution<double> dist(0, 1);
double lambdaPenalty = 50.0;

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
            value = value + w[i] * black_scholes_call(S, book.K[i], book.r[i], book.sigma[i], book.tau[i]);
        } else {
            value = value + w[i] * black_scholes_put(S, book.K[i], book.r[i], book.sigma[i], book.tau[i]);
        }
    }

    return value;
}

// Draw numPaths simulated end-of-horizon stock prices from the current RNG
// state. Called once per batch (training, out-of-sample, ...) so every
// objective/gradient evaluation against that batch sees the exact same
// paths, instead of resampling (and re-taping) fresh noise every call.
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
        + lambdaPenalty * returnViolation * returnViolation
        + lambdaPenalty * budgetViolation * budgetViolation;
}

// Gradient of penalized_objective w.r.t. w, via one reverse-mode AD pass.
std::vector<double> compute_gradient(
    const std::vector<double>& w, const std::vector<double>& paths, const OptionsBook& book,
    double valueTarget
) {
    g_tape<double>.reset();

    std::vector<Adjoint<double>> w_a;
    for (double wi : w) w_a.push_back(Adjoint<double>(wi));

    Adjoint<double> obj = penalized_objective(w_a, paths, book, valueTarget);
    g_tape<double>.init_adjoints();
    g_tape<double>.seed_adjoint(obj.idx, 1.0);
    g_tape<double>.propagate();

    std::vector<double> grad(w.size());
    for (size_t j = 0; j < w.size(); j++) grad[j] = g_tape<double>.get_adjoint(w_a[j].idx);
    return grad;
}

std::vector<double> vec_sub_scaled(const std::vector<double>& w, double step, const std::vector<double>& grad) {
    std::vector<double> result(w.size());
    for (size_t i = 0; i < w.size(); ++i)
        result[i] = w[i] - step * grad[i];
    return result;
}

double squared_norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return s;
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

    std::vector<std::vector<double>> grads;
    std::vector<double> means, variances;

    std::vector<double> trainPaths = generate_paths(market, simIter);

    for (int i = 0; i < startingPoints.size(); i++) {
        auto& w = startingPoints[i];
        auto& fw = finalWeights[i];
        std::cout << "Started at (" << w[0] << "," << w[1] << "," << w[2] << "," << w[3] << ")\n";

        stepSize = 0.0005;

        for (int iter = 0; iter < maxIter; iter++) {
            std::vector<double> grad = compute_gradient(w, trainPaths, book, valueTarget);
            double baseObjective = penalized_objective(w, trainPaths, book, valueTarget);

            // If the step size satisfies the Armijo condition after it grows, grow it
            while (true) {
                double candidateStepSize = stepSize * 1.5;
                std::vector<double> wCandidate = vec_sub_scaled(w, candidateStepSize, grad);

                double LHS = penalized_objective(wCandidate, trainPaths, book, valueTarget);
                double RHS = baseObjective - c * candidateStepSize * squared_norm(grad);

                if (LHS <= RHS) stepSize = candidateStepSize;
                else break;
            }

            // Ensure Armijo condition is fulfilled
            while (true) {
                std::vector<double> wCandidate = vec_sub_scaled(w, stepSize, grad);

                double LHS = penalized_objective(wCandidate, trainPaths, book, valueTarget);
                double RHS = baseObjective - c * stepSize * squared_norm(grad);

                if (LHS <= RHS) break;
                stepSize /= 2.0;
            }

            w = vec_sub_scaled(w, stepSize, grad);
            fw = w;

            if (squared_norm(grad) < gradTol) break;
        }

        std::cout << "  -> converged to (" << fw[0] << "," << fw[1] << "," << fw[2] << "," << fw[3] << ")\n";

        double weightSum = sum(fw);

        auto [finalMean, finalVariance] = portfolio_risk_return<double>(fw, trainPaths, book);
        std::vector<double> finalGrad = compute_gradient(fw, trainPaths, book, valueTarget);

        grads.push_back(finalGrad);
        means.push_back(finalMean);
        variances.push_back(finalVariance);
    }

    double relTol = 0.02;
    for (size_t k = 0; k < startingPoints.size(); k++) {
        double gradNorm = std::sqrt(squared_norm(grads[k]));
        bool wellConverged = gradNorm < gradTol;

        std::cout << "\nStart " << k << ": (";
        for (size_t j = 0; j < startingPoints[k].size(); j++)
            std::cout << startingPoints[k][j] << (j+1 < startingPoints[k].size() ? ", " : "");
        std::cout << ")\n";
        std::cout << "  Final weights: (";
        for (size_t j = 0; j < finalWeights[k].size(); j++)
            std::cout << finalWeights[k][j] << (j+1 < finalWeights[k].size() ? ", " : "");
        std::cout << ")\n";
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
