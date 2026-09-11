#pragma once

#include <cmath>
#include <limits>
#include <vector>


template <typename T, typename F>
std::vector<T> cfd_gradient(F&& f, std::vector<T>& inputs) {
    std::vector<T> grads;

    for (size_t i = 0; i < inputs.size(); i++) {
        T orig = inputs[i];
        // Scale the perturbation
        T hi = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), std::fabs(orig));

        inputs[i] = orig + hi; T fp = f(inputs);
        inputs[i] = orig - hi; T fm = f(inputs);
        inputs[i] = orig;
        
        grads.push_back((fp - fm) / (2 * hi));        
    }

    return grads;
}