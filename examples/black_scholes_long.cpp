#include <vector>
#include <random>
#include <iostream>

#include "adjoint.hpp"
#include "tangent.hpp"
#include "black_scholes_common.hpp"
#include "Eigen/Dense"


const int simIter = 500;
const double lambdaPenalty = 5.0;

Eigen::MatrixXd compute_hessian(
    const std::vector<double>& w,
    const std::vector<double>& paths,
    const OptionsBook& book,
    double valueTarget,
    bool longOnly=false
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

        Adjoint<Tangent<double>> obj = longOnly ? 
            penalized_objective_long(w_a, paths, book, valueTarget, lambdaPenalty) :
            penalized_objective(w_a, paths, book, valueTarget, lambdaPenalty);

        g_tape<Tangent<double>>.init_adjoints();
        g_tape<Tangent<double>>.seed_adjoint(obj.idx, Tangent<double>(1.0, 0.0));
        g_tape<Tangent<double>>.propagate();

        // Exploit symmetry
        for (int row = col; row < n; row++) {
            Tangent<double> adj = g_tape<Tangent<double>>.get_adjoint(w_a[row].idx);
            H(row, col) = adj.tangent;
            if (col != row) H(col, row) = adj.tangent;
        }
    }

    return H;
}


Eigen::VectorXd compute_gradient(
    const std::vector<double>& w,
    const std::vector<double>& paths,
    const OptionsBook& book,
    double valueTarget,
    bool longOnly=false
) {
    g_tape<double>.reset();

    std::vector<Adjoint<double>> w_a;
    for (double wi : w) w_a.push_back(Adjoint<double>(wi));

    Adjoint<double> obj = longOnly ? 
        penalized_objective_long(w_a, paths, book, valueTarget, lambdaPenalty) :
        penalized_objective(w_a, paths, book, valueTarget, lambdaPenalty);
    g_tape<double>.init_adjoints();
    g_tape<double>.seed_adjoint(obj.idx, 1.0);
    g_tape<double>.propagate();

    Eigen::VectorXd grad(w.size());
    for (size_t j = 0; j < w.size(); j++) grad(j) = g_tape<double>.get_adjoint(w_a[j].idx);
    return grad;
}

std::vector<double> toVec(const Eigen::VectorXd& v) {
    return std::vector<double>(v.data(), v.data() + v.size());
}

double squared_norm(const Eigen::VectorXd& v) {
    return v.squaredNorm();
}


int main(void) {
    std::vector<double> w = {0.25, 0.25, 0.25, 0.25};

    std::vector<double> trainPaths = generate_paths(market, simIter);

    Eigen::MatrixXd H = compute_hessian(w, trainPaths, book, valueTarget, true);
    Eigen::VectorXd grad = compute_gradient(w, trainPaths, book, valueTarget, true);

    Eigen::VectorXd w_v = Eigen::Map<const Eigen::VectorXd>(w.data(), w.size());

    for (int i = 0; i < 100; i++) {
        std::vector<double> wc = toVec(w_v);
        Eigen::VectorXd g = compute_gradient(wc, trainPaths, book, valueTarget);
        std::cout << "iter " << i << "  |g|=" << g.norm() << "  min w=" << w_v.minCoeff() << "\n";
        if (g.norm() < 1e-8) break;
        Eigen::MatrixXd H = compute_hessian(wc, trainPaths, book, valueTarget);
        w_v -= H.ldlt().solve(g);
    }

    // Pin and resolve to enforce w >=0 constraint
    int n = w_v.size();
    std::vector<bool> pinned(n, false);
    for (size_t i = 0; i < n; i++) {
        if (w_v(i) < 0) pinned[i] = true;
    }

    for (int pass = 0; pass < 100; pass++) {
        std::vector<size_t> freeIdx;
        for (size_t i = 0; i < n; i++) {
            if (pinned[i]) w_v(i) = 0.0;
            else freeIdx.push_back(i);
        }

        std::vector<double> wc = toVec(w_v);
        Eigen::VectorXd g = compute_gradient(wc, trainPaths, book, valueTarget, false);
        Eigen::MatrixXd H = compute_hessian(wc, trainPaths, book, valueTarget, false);

        Eigen::MatrixXd H_ff(freeIdx.size(), freeIdx.size());
        Eigen::VectorXd g_f(freeIdx.size());
        for (size_t a = 0; a < freeIdx.size(); a++) {
            g_f(a) = g(freeIdx[a]);
            for (size_t b = 0; b < freeIdx.size(); b++) H_ff(a, b) = H(freeIdx[a], freeIdx[b]);
        }
        Eigen::VectorXd delta = H_ff.ldlt().solve(g_f);
        for (size_t a = 0; a < freeIdx.size(); a++) w_v(freeIdx[a]) -= delta(a);

        bool changed = false;
        for (size_t i : freeIdx)
            if (w_v(i) < 0) { pinned[i] = true; changed = true; }     // free weight went negative: pin it

        if (!changed) {
            Eigen::VectorXd g2 = compute_gradient(toVec(w_v), trainPaths, book, valueTarget, false);
            for (size_t i = 0; i < n; i++)
                if (pinned[i] && g2(i) < 0) { pinned[i] = false; changed = true; }   // wall weight wants to be positive: release it
        }

        std::cout << "pass " << pass << "  pinned:";
        for (size_t i = 0; i < n; i++) if (pinned[i]) std::cout << " " << i;
        std::cout << "\n";
        if (!changed) break;
    }

    std::vector<double> w_final = toVec(w_v);

    double weightSum = sum(w_final);

    // Evaluate on the same frozen batch the Newton step converged against,
    // so this reflects the true state of the objective we optimized.
    auto [finalMean, finalVariance] = portfolio_risk_return<double>(w_final, trainPaths, book);
    Eigen::VectorXd finalGrad = compute_gradient(w_final, trainPaths, book, valueTarget);

    double constraintTol = 1e-3;
    bool constraintsOk = std::fabs(weightSum - 1.0) < constraintTol
                       && std::fabs(finalMean - valueTarget) < constraintTol;

    std::cout << "\n[Constraint check]\n";
    std::cout << "  Sum of weights:      " << weightSum << "   (target: 1.0,  error: " << std::fabs(weightSum - 1.0) << ")\n";
    std::cout << "  Expected portfolio value (simulated): " << finalMean
               << "   (target: " << valueTarget << ",  error: " << std::fabs(finalMean - valueTarget) << ")\n";
    std::cout << "  Constraints approximately satisfied? " << (constraintsOk ? "YES" : "NO") << "\n";

    std::cout << "\n[Stationarity check]\n";
    std::cout << "  Final gradient: " << toVec(finalGrad) << "\n";
    std::cout << "  Gradient norm:  " << std::sqrt(squared_norm(finalGrad)) << "   (should be small if converged)\n";

    std::cout << "\n[Result summary]\n";
    std::cout << "  Final weights: " << w_final << "\n";
    std::cout << "  Portfolio variance (risk): " << finalVariance << "\n";
    std::cout << "  Portfolio std dev:         " << std::sqrt(finalVariance) << "\n";
    std::cout << "  Expected value @ horizon:  " << finalMean << "\n";

    // Out-of-sample check: re-evaluate at w_final on a large, freshly-drawn
    // batch (engine continues on from wherever trainPaths left it, so this
    // is guaranteed to be independent of the training batch) to confirm the
    // fit generalizes rather than having exploited noise specific to it.
    std::vector<double> oosPaths = generate_paths(market, 50000);
    auto [oosMean, oosVariance] = portfolio_risk_return<double>(w_final, oosPaths, book);

    std::cout << "\n[Out-of-sample check, fresh " << oosPaths.size() << "-path batch]\n";
    std::cout << "  Expected value: " << oosMean << " (target " << valueTarget << ")\n";
    std::cout << "  Variance:       " << oosVariance << "\n";
}
