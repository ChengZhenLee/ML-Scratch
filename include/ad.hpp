#pragma once

#include <vector>


template <typename T>
struct Tape {
    struct Node {
        std::vector<int> parents;
        std::vector<T> derivatives;
        T adjoint = T(0);
    };

    std::vector<Node> nodes;

    int push_leaf() {
        // Push a leaf with no parents 
        nodes.push_back({{}, {}});
        return nodes.size() - 1;
    }

    int push_binary(int p1, int p2, T d1, T d2) {
        nodes.push_back({{p1, p2}, {d1, d2}});
        return nodes.size() - 1;
    }

    int push_unary(int p, T d) {
        nodes.push_back({{p}, {d}});
        return nodes.size() - 1;
    }

    void init_adjoints() {
        for (auto& node : nodes) node.adjoint = T(0);
    }

    void seed_adjoint(int idx, T value) {
        nodes[idx].adjoint += value;
    }

    void propagate() {
        for (int i = nodes.size() - 1; i >= 0; i--) {
            auto& curNode = nodes[i];
            auto& parents = curNode.parents;
            for (size_t j = 0; j < parents.size(); j++) {
                int p_idx = parents[j];
                nodes[p_idx].adjoint += curNode.derivatives[j] * curNode.adjoint;
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
struct Var {
    T value;
    int idx;

    Var(T v) : value(v), idx(g_tape<T>.push_leaf()) {}
    Var(T v, int i) : value(v), idx(i) {}
};

template <typename T>
Var<T> operator+(const Var<T>&a, const Var<T>&b) {
    T result_value = a.value + b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, 1.0, 1.0);
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> operator-(const Var<T>& a, const Var<T>& b) {
    T result_value = a.value - b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, 1.0, -1.0);
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> operator*(const Var<T>& a, const Var<T>& b) {
    T result_value = a.value * b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, b.value, a.value);
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> operator/(const Var<T>& a, const Var<T>& b) {
    T result_value = a.value / b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, 1.0 / b.value, -a.value / (b.value * b.value));
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> operator<(const Var<T>& a, const Var<T>& b) {
    return a.value < b.value;
}

template <typename T>
Var<T> operator<=(const Var<T>& a, const Var<T>& b) {
    return a.value <= b.value;
}

template <typename T>
Var<T> operator>(const Var<T>& a, const Var<T>& b) {
    return a.value > b.value;
}

template <typename T>
Var<T> operator>=(const Var<T>& a, const Var<T>& b) {
    return a.value >= b.value;
}

template <typename T>
Var<T> operator==(const Var<T>& a, const Var<T>& b) {
    return a.value == b.value;
}

template <typename T>
Var<T> exp(const Var<T>& x) {
    T result_value = std::exp(x.value);
    T result_idx = g_tape<T>.push_unary(x.idx, result_value);
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> sin(const Var<T>& x) {
    T result_value = std::sin(x.value);
    T result_idx = g_tape<T>.push_unary(x.idx, std::cos(x.value));
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> cos(const Var<T>& x) {
    T result_value = std::cos(x.value);
    T result_idx = g_tape<T>.push_unary(x.idx, -std::sin(x.value));
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> tan(const Var<T>& x) {
    T result_value = std::tan(x.value);
    T result_idx = g_tape<T>.push_unary(x.idx, 1 / (std::cos(x.value) * std::cos(x.value)));
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> log(const Var<T>& x) {
    T result_value = std::log(x.value);
    T result_idx = g_tape<T>.push_unary(x.idx, 1 / x.value);
    return Var<T>(result_value, result_idx);
}