#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "black_scholes_common.hpp"


int maxIter = 8000;
int simIter = 500;
double stepSize = 0.0005;
double c = 1e-4;
double h = 1e-4;
double gradTol = 1e-4;
double lambdaPenalty = 50.0;

// Gradient of penalized_objective w.r.t. w, via central finite differences.
std::vector<double> compute_gradient(
    std::vector<double>& w, const std::vector<double>& paths, const OptionsBook& book,
    double valueTarget
) {
    std::vector<double> grad;

    for (size_t i = 0; i < w.size(); i++) {
        double orig = w[i];
        w[i] = orig - h;
        double fMinus = penalized_objective(w, paths, book, valueTarget, lambdaPenalty);
        w[i] = orig + h;
        double fPlus = penalized_objective(w, paths, book, valueTarget, lambdaPenalty);
        w[i] = orig;
        grad.push_back((fPlus - fMinus) / (2 * h));
    }

    return grad;
}


int main(void) {
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
        double baseObjective = penalized_objective(w, trainPaths, book, valueTarget, lambdaPenalty);

        // If the step size satisfies the Armijo condition after it grows, grow it
        while (true) {
            double candidateStepSize = stepSize * 1.5;
            std::vector<double> wCandidate = vec_sub_scaled(w, candidateStepSize, grad);

            double LHS = penalized_objective(wCandidate, trainPaths, book, valueTarget, lambdaPenalty);
            double RHS = baseObjective - c * candidateStepSize * squared_norm(grad);

            if (LHS <= RHS) stepSize = candidateStepSize;
            else break;
        }

        // Ensure Armijo condition is fulfilled
        while (true) {
            std::vector<double> wCandidate = vec_sub_scaled(w, stepSize, grad);

            double LHS = penalized_objective(wCandidate, trainPaths, book, valueTarget, lambdaPenalty);
            double RHS = baseObjective - c * stepSize * squared_norm(grad);

            if (LHS <= RHS) break;
            stepSize /= 2.0;
        }

        w = vec_sub_scaled(w, stepSize, grad);

        if (squared_norm(grad) < gradTol) break;
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
