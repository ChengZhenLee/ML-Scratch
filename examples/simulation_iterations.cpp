#include <iostream>
#include <random>

#include "black_scholes.hpp"


const int numTrials = 20;
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
            value = value + w[i] * black_scholes_call(S, book.K[i], book.r[i], book.sigma[i], book.tau[i]);
        } else {
            value = value + w[i] * black_scholes_put(S, book.K[i], book.r[i], book.sigma[i], book.tau[i]);
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


int main(void) {
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

    std::vector<int> simIters = {100, 500, 1000, 5000, 20000, 50000};

    std::vector<double> means;
    std::vector<double> variances;

    for (auto simIter : simIters) {
        std::vector<double> trialVariances;

        // Calculate variances across multiple batches
        for (int i = 0; i < numTrials; i++) {
            std::vector<double>paths = generate_paths(market, simIter);
            auto [_, variance] = portfolio_risk_return(w, paths, book);
            trialVariances.push_back(variance);
        }

        // Calculate the spread of the variance (variance of variance)
        double avgVariance = 0.0;
        for (double v : trialVariances) {
            avgVariance += v;
        }
        avgVariance /= numTrials;

        double spreadSqSum = 0.0;
        for (double v : trialVariances) {
            spreadSqSum += (v - avgVariance) * (v - avgVariance);
        }

        // Sample variance (n - 1) 
        double spreadVar = spreadSqSum / (numTrials - 1);
        double spread = std::sqrt(spreadVar);

        std::cout << "simIter=" << simIter
            << "  avg variance estimate=" << avgVariance
            << "  spread across " << numTrials << " trials=" << spread << "\n";
    }

    return 0;
}