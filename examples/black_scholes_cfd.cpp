#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "black_scholes.hpp"


int maxIter = 8000;
int simIter = 500;
double stepSize = 0.0005;
double c = 1e-4;
double h = 1e-4;
std::mt19937_64 engine;
std::normal_distribution<double> dist(0, 1);
double lambdaPenalty = 500.0;

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
// paths, instead of resampling fresh noise every call.
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

// Gradient of penalized_objective w.r.t. w, via central finite differences.
std::vector<double> compute_gradient(
    std::vector<double>& w, const std::vector<double>& paths, const OptionsBook& book,
    double valueTarget
) {
    std::vector<double> grad;

    for (size_t i = 0; i < w.size(); i++) {
        double orig = w[i];
        w[i] = orig - h;
        double fMinus = penalized_objective(w, paths, book, valueTarget);
        w[i] = orig + h;
        double fPlus = penalized_objective(w, paths, book, valueTarget);
        w[i] = orig;
        grad.push_back((fPlus - fMinus) / (2 * h));
    }

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

    std::vector<double> w = {0.25, 0.25, 0.25, 0.25};

    // Freeze one batch of simulated price paths for the whole optimization,
    // turning the Monte Carlo objective into a fixed, deterministic function
    // of w (a sample average approximation) instead of a fresh noisy draw
    // every iteration. Without this, each iteration compared against a
    // differently-noisy batch, which broke the Armijo line search's
    // sufficient-decrease test and collapsed stepSize to near zero within
    // the first few iterations.
    std::vector<double> trainPaths = generate_paths(market, simIter);

    for (int i = 0; i < maxIter; i++) {
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
    }

    double weightSum = sum(w);

    // Evaluate on the same frozen batch the optimizer converged against,
    // so this reflects the true state of the objective we optimized.
    auto [finalMean, finalVariance] = portfolio_risk_return<double>(w, trainPaths, book);
    std::vector<double> finalGrad = compute_gradient(w, trainPaths, book, valueTarget);

    double constraintTol = 1e-3;
    bool constraintsOk = std::fabs(weightSum - 1.0) < constraintTol
                       && std::fabs(finalMean - valueTarget) < constraintTol;

    std::cout << "\n[Constraint check]\n";
    std::cout << "  Sum of weights:      " << weightSum << "   (target: 1.0,  error: " << std::fabs(weightSum - 1.0) << ")\n";
    std::cout << "  Expected portfolio value (simulated): " << finalMean
               << "   (target: " << valueTarget << ",  error: " << std::fabs(finalMean - valueTarget) << ")\n";
    std::cout << "  Constraints approximately satisfied? " << (constraintsOk ? "YES" : "NO") << "\n";

    std::cout << "\n[Stationarity check]\n";
    std::cout << "  Final gradient: " << finalGrad << "\n";
    std::cout << "  Gradient norm:  " << std::sqrt(squared_norm(finalGrad)) << "   (should be small if converged)\n";

    std::cout << "\n[Result summary]\n";
    std::cout << "  Final weights: " << w << "\n";
    std::cout << "  Portfolio variance (risk): " << finalVariance << "\n";
    std::cout << "  Portfolio std dev:         " << std::sqrt(finalVariance) << "\n";
    std::cout << "  Expected value @ horizon:  " << finalMean << "\n";

    // Out-of-sample check: re-evaluate at w on a large, freshly-drawn batch
    // (engine continues on from wherever trainPaths left it, so this is
    // guaranteed to be independent of the training batch) to confirm the
    // fit generalizes rather than having exploited noise specific to it.
    std::vector<double> oosPaths = generate_paths(market, 50000);
    auto [oosMean, oosVariance] = portfolio_risk_return<double>(w, oosPaths, book);

    std::cout << "\n[Out-of-sample check, fresh " << oosPaths.size() << "-path batch]\n";
    std::cout << "  Expected value: " << oosMean << " (target " << valueTarget << ")\n";
    std::cout << "  Variance:       " << oosVariance << "\n";
}
