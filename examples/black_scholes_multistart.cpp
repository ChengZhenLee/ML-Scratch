#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "adjoint.hpp"
#include "black_scholes_common.hpp"


int maxIter = 8000;
int simIter = 500;
double stepSize = 0.0005;
double c = 1e-4;
double gradTol = 1e-4;
double lambdaPenalty = 50.0;

// Gradient of penalized_objective w.r.t. w, via one reverse-mode AD pass.
std::vector<double> compute_gradient(
    const std::vector<double>& w, const std::vector<double>& paths, const OptionsBook& book,
    double valueTarget
) {
    g_tape<double>.reset();

    std::vector<Adjoint<double>> w_a;
    for (double wi : w) w_a.push_back(Adjoint<double>(wi));

    Adjoint<double> obj = penalized_objective(w_a, paths, book, valueTarget, lambdaPenalty);
    g_tape<double>.init_adjoints();
    g_tape<double>.seed_adjoint(obj.idx, 1.0);
    g_tape<double>.propagate();

    std::vector<double> grad(w.size());
    for (size_t j = 0; j < w.size(); j++) grad[j] = g_tape<double>.get_adjoint(w_a[j].idx);
    return grad;
}


int main(void) {
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
