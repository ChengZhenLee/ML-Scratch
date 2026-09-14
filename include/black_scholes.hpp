#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include "adjoint.hpp"


template <typename T>
T norm_cdf(T x) {
    return T(1) / T(2) *  (T(1) + erf(x / sqrt(T(2))));
}

template <typename T>
T norm_pdf(T x) {
    return T(1) / sqrt(T(2) * T(M_PI)) * exp(-pow(x, 2) / T(2));
}

template <typename T>
T dp(T S, T K, T r, T sigma, T tau) {
    return (log(S/K) + (r + pow(sigma, 2) / T(2)) * tau) / (sigma * sqrt(tau));
}

template <typename T>
T dm(T S, T K, T r, T sigma, T tau) {
    return dp(S, K, r, sigma, tau) - sigma * sqrt(tau);
}

template <typename T>
T black_scholes_call(T S, T K, T r, T sigma, T tau) {
    T dpResult = dp(S, K, r, sigma, tau);
    T dmResult = dm(S, K, r, sigma, tau);
    return norm_cdf(dpResult) * S - norm_cdf(dmResult) * K * exp(-r * tau);
}

template <typename T>
T black_scholes_put(T S, T K, T r, T sigma, T tau) {
    T dpResult = dp(S, K, r, sigma, tau);
    T dmResult = dm(S, K, r, sigma, tau);
    return norm_cdf(-dmResult) * K * exp(-r * tau) - norm_cdf(-dpResult) * S;
}

template <typename T>
T delta_call(T S, T K, T r, T sigma, T tau) {
    return norm_cdf(dp(S, K, r, sigma, tau));
}

template <typename T>
T delta_put(T S, T K, T r, T sigma, T tau) {
    return norm_cdf(dp(S, K, r, sigma, tau)) - T(1);
}

template <typename T>
T vega(T S, T K, T r, T sigma, T tau) {
    T d1 = dp(S, K, r, sigma, tau);
    return S * norm_pdf(d1) * sqrt(tau);
}

template <typename T>
T theta_call(T S, T K, T r, T sigma, T tau) {
    T d1 = dp(S, K, r, sigma, tau);
    T d2 = dm(S, K, r, sigma, tau);
    return -(S * norm_pdf(d1) * sigma) / (T(2) * sqrt(tau)) - r * K * exp(-r * tau) * norm_cdf(d2);
}

template <typename T>
T rho_call(T S, T K, T r, T sigma, T tau) {
    T d2 = dm(S, K, r, sigma, tau);
    return K * tau * exp(- r * tau) * norm_cdf(d2);
}

template <typename T>
T gamma(T S, T K, T r, T sigma, T tau) {
    T d1 = dp(S, K, r, sigma, tau);
    return norm_pdf(d1) / (S * sigma * sqrt(tau));
}

template <typename T>
T vanna(T S, T K, T r, T sigma, T tau) {
    T d1 = dp(S, K, r, sigma, tau);
    T d2 = dm(S, K, r, sigma, tau);
    return -norm_pdf(d1) * d2 / sigma;
}

template <typename T>
T charm(T S, T K, T r, T sigma, T tau) {
    T d1 = dp(S, K, r, sigma, tau);
    return -norm_pdf(d1) * ((r + sigma*sigma/T(2)) / (sigma*sqrt(tau)) - d1/(T(2)*tau));
}