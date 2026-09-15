#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <vector>


template <typename T>
struct Tape {
    // Every node produced by this AD system has at most 2 parents
    // (binary ops); fixed-size storage avoids a heap allocation per
    // operation that std::vector members would incur.
    struct Node {
        int parents[2] = {0, 0};
        T derivatives[2] = {T(0), T(0)};
        uint8_t numParents = 0;
        T adjoint = T(0);
    };

    std::vector<Node> nodes;

    int push_leaf() {
        // Push a leaf with no parents
        nodes.push_back(Node{});
        return nodes.size() - 1;
    }

    int push_binary(int p1, int p2, T d1, T d2) {
        Node node{};
        node.parents[0] = p1;
        node.parents[1] = p2;
        node.derivatives[0] = d1;
        node.derivatives[1] = d2;
        node.numParents = 2;
        nodes.push_back(node);
        return nodes.size() - 1;
    }

    int push_unary(int p, T d) {
        Node node{};
        node.parents[0] = p;
        node.derivatives[0] = d;
        node.numParents = 1;
        nodes.push_back(node);
        return nodes.size() - 1;
    }

    void init_adjoints() {
        for (auto& node : nodes) node.adjoint = T(0);
    }

    void seed_adjoint(int idx, T value) {
        nodes[idx].adjoint = nodes[idx].adjoint + value;
    }

    void propagate() {
        for (int i = nodes.size() - 1; i >= 0; i--) {
            auto& curNode = nodes[i];
            for (uint8_t j = 0; j < curNode.numParents; j++) {
                int p_idx = curNode.parents[j];
                nodes[p_idx].adjoint = nodes[p_idx].adjoint + curNode.derivatives[j] * curNode.adjoint;
            }
        }
    }

    T get_adjoint(int idx) {
        if (idx < 0 || idx >= nodes.size()) return T(0);
        return nodes[idx].adjoint;
    }

    void reset() {
        nodes.clear();
    }
};

template <typename T>
inline Tape<T> g_tape;

template <typename T>
struct Adjoint {
    T value;
    int idx;

    Adjoint(T v) : value(v), idx(g_tape<T>.push_leaf()) {}
    Adjoint(T v, int i) : value(v), idx(i) {}
};

template <typename T>
Adjoint<T> operator+(const Adjoint<T>&a, const Adjoint<T>&b) {
    T result_value = a.value + b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, T(1), T(1));
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> operator+(const Adjoint<T>&a, const T&b) {
    T result_value = a.value + b;
    int result_idx = g_tape<T>.push_unary(a.idx, T(1));
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> operator+(const T& a, const Adjoint<T>& b) {
    return b + a;
}

template <typename T>
Adjoint<T> operator-(const Adjoint<T>& a, const Adjoint<T>& b) {
    T result_value = a.value - b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, T(1), -T(1));
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> operator-(const Adjoint<T>& a, const T& b) {
    T result_value = a.value - b;
    int result_idx = g_tape<T>.push_unary(a.idx, T(1));
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> operator-(const T& a, const Adjoint<T>& b) {
    T result_value = a - b.value;
    int result_idx = g_tape<T>.push_unary(b.idx, -T(1));
    return Adjoint<T>(result_value, result_idx);
}

template<typename T>
Adjoint<T> operator-(const Adjoint<T>& x) {
    T result_value = -x.value;
    int result_idx = g_tape<T>.push_unary(x.idx, T(-1));
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> operator*(const Adjoint<T>& a, const Adjoint<T>& b) {
    T result_value = a.value * b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, b.value, a.value);
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> operator*(const Adjoint<T>& a, const T& b) {
    T result_value = a.value * b;
    int result_idx = g_tape<T>.push_unary(a.idx, b);
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> operator*(const T& a, const Adjoint<T>& b) {
    return b * a;
}

template <typename T>
Adjoint<T> operator/(const Adjoint<T>& a, const Adjoint<T>& b) {
    T result_value = a.value / b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, 1.0 / b.value, -a.value / (b.value * b.value));
    return Adjoint<T>(result_value, result_idx);
}

template<typename T>
Adjoint<T> operator/(const Adjoint<T>& a, const T& b) {
    T result_value = a.value / b;
    int result_idx = g_tape<T>.push_unary(a.idx, T(1)/b);
    return Adjoint<T>(result_value, result_idx);
}

template<typename T>
Adjoint<T> operator/(const T& a, const Adjoint<T>& b) {
    T result_value = a / b.value;
    T local_partial = -a / (b.value * b.value);   // d/db [a/b] = -a/b^2
    int result_idx = g_tape<T>.push_unary(b.idx, local_partial);
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
bool operator<(const Adjoint<T>& a, const Adjoint<T>& b) {
    return a.value < b.value;
}

template <typename T>
bool operator<(const Adjoint<T>& a, const T& b) {
    return a.value < b;
}

template <typename T>
bool operator<=(const Adjoint<T>& a, const Adjoint<T>& b) {
    return a.value <= b.value;
}

template <typename T>
bool operator<=(const Adjoint<T>& a, const T& b) {
    return a.value <= b;
}

template <typename T>
bool operator>(const Adjoint<T>& a, const Adjoint<T>& b) {
    return a.value > b.value;
}

template <typename T>
bool operator>(const Adjoint<T>& a, const T& b) {
    return a.value > b;
}

template <typename T>
bool operator>=(const Adjoint<T>& a, const Adjoint<T>& b) {
    return a.value >= b.value;
}

template <typename T>
bool operator>=(const Adjoint<T>& a, const T& b) {
    return a.value >= b;
}

template <typename T>
bool operator<(const T&a, const Adjoint<T>& b) {
    return b > a;
}

template <typename T>
bool operator<=(const T&a, const Adjoint<T>& b) {
    return b >= a;
}

template <typename T>
bool operator>(const T&a, const Adjoint<T>& b) {
    return b < a;
}

template <typename T>
bool operator>=(const T&a, const Adjoint<T>& b) {
    return b <= a;
}

template <typename T>
bool operator==(const Adjoint<T>& a, const Adjoint<T>& b) {
    return a.value == b.value;
}

template <typename T>
bool operator==(const Adjoint<T>& a, const T& b) {
    return a.value == b;
}

template <typename T>
bool operator==(const T& a, const Adjoint<T>& b) {
    return b == a;
}

template <typename T>
Adjoint<T> exp(const Adjoint<T>& x) {
    T result_value = exp(x.value);
    int result_idx = g_tape<T>.push_unary(x.idx, result_value);
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> sin(const Adjoint<T>& x) {
    T result_value = sin(x.value);
    int result_idx = g_tape<T>.push_unary(x.idx, cos(x.value));
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> cos(const Adjoint<T>& x) {
    T result_value = cos(x.value);
    int result_idx = g_tape<T>.push_unary(x.idx, -sin(x.value));
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> tan(const Adjoint<T>& x) {
    T result_value = tan(x.value);
    int result_idx = g_tape<T>.push_unary(x.idx, T(1) / (cos(x.value) * cos(x.value)));
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> log(const Adjoint<T>& x) {
    T result_value = log(x.value);
    int result_idx = g_tape<T>.push_unary(x.idx, T(1) / x.value);
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> pow(const Adjoint<T>& x, int n) {
    T result_value = pow(x.value, n);
    T local_partial = T(n) * pow(x.value, n - 1);
    int result_idx = g_tape<T>.push_unary(x.idx, local_partial);
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> sqrt(const Adjoint<T>& x) {
    T result_value = sqrt(x.value);
    T local_partial = T(1) / (T(2) * result_value);
    int result_idx = g_tape<T>.push_unary(x.idx, local_partial);
    return Adjoint<T>(result_value, result_idx);
}

template <typename T>
Adjoint<T> erf(const Adjoint<T>& x) {
    T result_value = erf(x.value);
    T local_partial = (T(2) / sqrt(T(M_PI))) * exp(-x.value * x.value);
    int result_idx = g_tape<T>.push_unary(x.idx, local_partial);
    return Adjoint<T>(result_value, result_idx);
}