#pragma once

// Setup shared by the black_scholes_* examples: the market / options book
// being optimized, the Monte Carlo path generator, the penalized objective,
// and a few small vector helpers.

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "black_scholes.hpp"


inline std::mt19937_64 engine;
inline std::normal_distribution<double> dist(0, 1);

// Per-underlying market parameters shared by every option in the book.
struct MarketParams {
    double S0, mu, sigmaStock, horizon;
};

// Fixed parameters of the options
struct OptionsBook {
    std::vector<double> K, r, sigma, tau;
    std::vector<int> isCall;
};

// Use same underlying stock for all options
inline const MarketParams market{100.0, 0.08, 0.22, 1.0 / 52.0};   // S0, mu, sigmaStock, horizon (one week ahead)
inline const double valueTarget = 10.0;

inline const OptionsBook book{
    {95.0,  105.0, 95.0,  105.0},   // K
    {0.05,  0.05,  0.05,  0.05},    // r
    {0.20,  0.20,  0.25,  0.25},    // sigma
    {0.5,   0.5,   1.0,   1.0},     // tau
    {1,     1,     0,     0},       // isCall
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

// Draw numPaths simulated end-of-horizon stock prices from the current RNG
// state. Called once per batch (training, out-of-sample, ...) so every
// objective/gradient evaluation against that batch sees the exact same
// paths, instead of resampling (and re-taping) fresh noise every call.
inline std::vector<double> generate_paths(const MarketParams& market, int numPaths) {
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
T relu(const T& x) {
    return (x > T(0)) ? x : T(0);
}

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
    double valueTarget, double lambda
) {
    auto result = portfolio_risk_return(w, paths, book);

    T weightSum = T(0.0);
    for (const auto& wi : w) weightSum = weightSum + wi;

    T returnViolation = result.mean - T(valueTarget);
    T budgetViolation = weightSum - T(1.0);

    return result.variance
        + T(lambda) * returnViolation * returnViolation
        + T(lambda) * budgetViolation * budgetViolation;
}

template <typename T>
T penalized_objective_long(const std::vector<T>& w, const std::vector<double>& paths, const OptionsBook& book,
    double valueTarget, double lambda
) {
    T obj = penalized_objective(w, paths, book, valueTarget, lambda);

    for (size_t i = 0; i < w.size(); i++) {
        T penalty = relu(-w[i]);
        obj = obj + T(lambda) * penalty * penalty;
    }

    return obj;
}

inline std::vector<double> vec_sub_scaled(const std::vector<double>& w, double step, const std::vector<double>& grad) {
    std::vector<double> result(w.size());
    for (size_t i = 0; i < w.size(); ++i)
        result[i] = w[i] - step * grad[i];
    return result;
}

inline double squared_norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return s;
}

inline double sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

// Overload the << operator for ostream to print out vectors
inline std::ostream& operator<<(std::ostream& os, const std::vector<double>& v) {
    os << "(";
    for (size_t j = 0; j < v.size(); j++) os << v[j] << (j + 1 < v.size() ? ", " : "");
    return os << ")";
}
