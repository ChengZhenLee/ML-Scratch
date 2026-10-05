#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

#include "adjoint.hpp"
#include "tangent.hpp"
#include "black_scholes_common.hpp"
#include "Eigen/Dense"


const int simIter = 500;

Eigen::MatrixXd compute_hessian(
    const std::vector<double>& w,
    const std::vector<double>& paths,
    const OptionsBook& book,
    double valueTarget, double lambda
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

        Adjoint<Tangent<double>> obj = penalized_objective(w_a, paths, book, valueTarget, lambda);

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
    double valueTarget, double lambda
) {
    g_tape<double>.reset();

    std::vector<Adjoint<double>> w_a;
    for (double wi : w) w_a.push_back(Adjoint<double>(wi));

    Adjoint<double> obj = penalized_objective(w_a, paths, book, valueTarget, lambda);
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
    // Sweep over different lambda_penalty values
    std::vector<double> lambdas = {1, 5, 10, 50, 100, 500, 2000, 10000};

    std::vector<double> w = {0.25, 0.25, 0.25, 0.25};

    std::vector<double> trainPaths = generate_paths(market, simIter);
    std::vector<double> oosPaths = generate_paths(market, 50000);

    std::cout << std::setw(8)  << "lambda"     << std::setw(12) << "|sum-1|"
          << std::setw(12) << "|mean-tgt|" << std::setw(14) << "2*lam*retErr"
          << std::setw(12) << "variance"   << std::setw(12) << "oosVar"
          << std::setw(12) << "gradAfter"  << std::setw(12) << "cond(H)" << "\n";

    for (double lambda : lambdas) {
        Eigen::MatrixXd H = compute_hessian(w, trainPaths, book, valueTarget, lambda);
        Eigen::VectorXd g = compute_gradient(w, trainPaths, book, valueTarget, lambda);

        Eigen::VectorXd w_v = Eigen::Map<const Eigen::VectorXd>(w.data(), w.size());
        Eigen::VectorXd w_star = w_v - H.ldlt().solve(g);
        std::vector<double> w_final = toVec(w_star);

        auto [mean, var] = portfolio_risk_return<double>(w_final, trainPaths, book);
        auto [oosMean, oosVar] = portfolio_risk_return<double>(w_final, oosPaths, book);
        double gradAfter = compute_gradient(w_final, trainPaths, book, valueTarget, lambda).norm();

        Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(H).eigenvalues();
        double retErr = mean - valueTarget;

        std::cout << std::setw(8)  << lambda << std::setw(12) << std::fabs(sum(w_final) - 1.0)
            << std::setw(12) << std::fabs(retErr) << std::setw(14) << 2 * lambda * retErr
            << std::setw(12) << var << std::setw(12) << oosVar
            << std::setw(12) << gradAfter << std::setw(12) << ev.maxCoeff() / ev.minCoeff() << "\n";
    }
}
