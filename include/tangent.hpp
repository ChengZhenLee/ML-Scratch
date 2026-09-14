#pragma once

#define _USE_MATH_DEFINES
#include <cmath>


template <typename T>
struct Tangent {
    T value;
    T tangent = T(0);

    Tangent(T value) : value(value) {}

    Tangent(T value, T tangent) :
        value(value), tangent(tangent) {}

    void seed_tangent(T seed) {
        tangent = seed;
    }
};


template <typename T>
Tangent<T> operator+(const Tangent<T>& a, const Tangent<T>& b) {
    T return_value = a.value + b.value;
    T tangent = a.tangent + b.tangent;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator+(const Tangent<T>& a, const T& b) {
    T return_value = a.value + b;
    T tangent = a.tangent;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator+(const T& a, const Tangent<T>& b) {
    return b + a;
}

template <typename T>
Tangent<T> operator-(const Tangent<T>&a, const Tangent<T>& b) {
    T return_value = a.value - b.value;
    T tangent = a.tangent - b.tangent;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator-(const Tangent<T>& a, const T& b) {
    T return_value = a.value - b;
    T tangent = a.tangent;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator-(const T& a, const Tangent<T>& b) {
    T return_value = a - b.value;
    T tangent = -b.tangent;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator-(const Tangent<T>& x) {
    T return_value = -x.value;
    T tangent = -x.tangent;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator*(const Tangent<T>& a, const Tangent<T>& b) {
    T return_value = a.value * b.value;
    T tangent = a.tangent * b.value + a.value * b.tangent;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator*(const Tangent<T>& a, const T& b) {
    T return_value = a.value * b;
    T tangent = a.tangent * b;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator*(const T& a, const Tangent<T>& b) {
    return b * a;
}

template <typename T>
Tangent<T> operator/(const Tangent<T>& a, const Tangent<T>& b) {
    T return_value = a.value / b.value;
    T tangent = a.tangent / b.value - (a.value * b.tangent) / (b.value * b.value);
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator/(const Tangent<T>& a, const T& b) {
    T return_value = a.value / b;
    T tangent = a.tangent / b;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> operator/(const T& a, const Tangent<T>& b) {
    T return_value = a / b.value;
    T tangent = - (a * b.tangent) / (b.value * b.value);
    return Tangent<T>(return_value, tangent);
}

template <typename T>
bool operator<(const Tangent<T>& a, const Tangent<T>& b) {
    return a.value < b.value;
}

template <typename T>
bool operator<(const Tangent<T>& a, const T& b) {
    return a.value < b;
}

template <typename T>
bool operator<=(const Tangent<T>& a, const Tangent<T>& b) {
    return a.value <= b.value;
}

template <typename T>
bool operator<=(const Tangent<T>& a, const T& b) {
    return a.value <= b;
}

template <typename T>
bool operator>(const Tangent<T>& a, const Tangent<T>& b) {
    return a.value > b.value;
}

template <typename T>
bool operator>(const Tangent<T>& a, const T& b) {
    return a.value > b;
}

template <typename T>
bool operator>=(const Tangent<T>& a, const Tangent<T>& b) {
    return a.value >= b.value;
}

template <typename T>
bool operator>=(const Tangent<T>& a, const T& b) {
    return a.value >= b;
}

template <typename T>
bool operator<(const T&a, const Tangent<T>& b) {
    return b > a;
}

template <typename T>
bool operator<=(const T&a, const Tangent<T>& b) {
    return b >= a;
}

template <typename T>
bool operator>(const T&a, const Tangent<T>& b) {
    return b < a;
}

template <typename T>
bool operator>=(const T&a, const Tangent<T>& b) {
    return b <= a;
}

template <typename T>
bool operator==(const Tangent<T>& a, const Tangent<T>& b) {
    return a.value == b.value;
}

template <typename T>
bool operator==(const Tangent<T>& a, const T& b) {
    return a.value == b;
}

template <typename T>
bool operator==(const T& a, const Tangent<T>& b) {
    return b == a;
}

template <typename T>
Tangent<T> exp(const Tangent<T>&x) {
    T return_value = std::exp(x.value);
    T tangent = x.tangent * return_value;
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> sin(const Tangent<T>&x) {
    T return_value = std::sin(x.value);
    T tangent = x.tangent * std::cos(x.value);
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> cos(const Tangent<T>& x) {
    T return_value = std::cos(x.value);
    T tangent = x.tangent * -std::sin(x.value);
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> tan(const Tangent<T>& x) {
    T return_value = std::tan(x.value);
    T tangent = x.tangent * 1 / (std::cos(x.value) * std::cos(x.value));
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> log(const Tangent<T>& x) {
    T return_value = std::log(x.value);
    T tangent = x.tangent * 1/ (x.value);
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> pow(const Tangent<T>& x, int n) {
    T return_value = std::pow(x.value, n);
    T tangent = x.tangent * T(n) * std::pow(x.value, n - 1);
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> sqrt(const Tangent<T>& x) {
    T return_value = std::sqrt(x.value);
    T tangent = x.tangent / (T(2) * std::sqrt(x.value));
    return Tangent<T>(return_value, tangent);
}

template <typename T>
Tangent<T> erf(const Tangent<T>& x) {
    T return_value = std::erf(x.value);
    T tangent = x.tangent * (T(2) / std::sqrt(T(M_PI))) * std::exp(-x.value * x.value);
    return Tangent<T>(return_value, tangent);
}