#include <benchmark/benchmark.h>
#include "adjoint.hpp"
#include "cfd.hpp"


template <typename T>
T test_function(const std::vector<T>& x) {
    T result(0.0);
    for (size_t i = 0; i < x.size(); i++) {
        result = result + exp(x[i]) * log(x[i] + T(2.0)) + sqrt(x[i]*x[i] + T(1.0));
    }
    return result;
}

std::vector<double> ad_gradient(const std::vector<double>& x) {
    g_tape<double>.reset();
    std::vector<Adjoint<double>> x_a;
    for (double v : x) x_a.push_back(Adjoint<double>(v));

    Adjoint<double> result = test_function(x_a);

    g_tape<double>.init_adjoints();
    g_tape<double>.seed_adjoint(result.idx, 1.0);
    g_tape<double>.propagate();

    std::vector<double> grad(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        grad[i] = g_tape<double>.get_adjoint(x_a[i].idx);
    }
    return grad;
}

static void BM_AD_GRADIENT(benchmark::State& state) {
    int n = state.range(0);
    std::vector<double> x(n);
    for (int i = 0; i < x.size(); i++) {
        x[i] = 1.0 + 0.1 * i;
    }

    for (auto _ : state) {
        auto grad = ad_gradient(x);
        benchmark::DoNotOptimize(grad);
    }
}
BENCHMARK(BM_AD_GRADIENT)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

std::vector<double> fd_gradient(std::vector<double>& x) {
    return cfd_gradient(
        [](std::vector<double>& x) { return test_function<double>(x); },
        x
    );
}

static void BM_FD_GRADIENT(benchmark::State& state) {
    int n = state.range(0);
    std::vector<double> x(n);
    for (int i = 0; i < x.size(); i++) {
        x[i] = 1.0 + 0.1 * i;
    }

    for (auto _ : state) {
        auto grad = fd_gradient(x);
        benchmark::DoNotOptimize(grad);
    }
}
BENCHMARK(BM_FD_GRADIENT)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

BENCHMARK_MAIN();