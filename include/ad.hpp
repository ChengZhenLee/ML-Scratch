#pragma once

#include <vector>


template <typename T>
struct Tape {
    struct Node {
        std::vector<int> parents;
        std::vector<T> derivatives;
    };

    std::vector<Node> nodes;
    std::vector<T> adjoints;

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

    void backward(int outputIndex) {
        this->adjoints.assign(this->nodes.size(), T(0));

        // Seed the (single) output
        this->adjoints[outputIndex] = T(1);

        for (int i = this->nodes.size() - 1; i >= 0; i--) {
            for (int j = 0; j < this->nodes[i].parents.size(); j++) {
                int parent = this->nodes[i].parents[j];
                this->adjoints[parent] += this->nodes[i].derivatives[j] * this->adjoints[i];
            }
        }
    }

    void reset() {
        this->nodes.clear();
        this->adjoints.clear();
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
    double result_value = a.value + b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, 1.0, 1.0);
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> operator-(const Var<T>& a, const Var<T>& b) {
    double result_value = a.value - b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, 1.0, -1.0);
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> operator*(const Var<T>& a, const Var<T>& b) {
    double result_value = a.value * b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, b.value, a.value);
    return Var<T>(result_value, result_idx);
}

template <typename T>
Var<T> operator/(const Var<T>& a, const Var<T>& b) {
    double result_value = a.value / b.value;
    int result_idx = g_tape<T>.push_binary(a.idx, b.idx, 1.0 / b.value, -a.value / (b.value * b.value));
    return Var<T>(result_value, result_idx);
}