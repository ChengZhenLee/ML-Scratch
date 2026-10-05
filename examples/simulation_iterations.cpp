#include <iostream>
#include <random>

#include "black_scholes_common.hpp"


const int numTrials = 20;


int main(void) {
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
