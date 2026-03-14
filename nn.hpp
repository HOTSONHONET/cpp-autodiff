#pragma once
#include "value.hpp"


inline std::mt19937 rng(42);
inline std::uniform_real_distribution<double> dist(-1.0, 1.0);
struct Neuron {
    vector<Value*> w;
    Value* b;

    Neuron(int nin) {
        for(int i = 0; i < nin; i++) {
            w.push_back(Value::create(dist(rng)));
        }
        b = Value::create(0.0);
    }

    Value* operator()(const vector<Value*>&x) {
        Value* out = Value::create(0.0);
        for(int i = 0; i < (int) x.size(); i++) {
            out = add(out, mul(w[i], x[i]));
        }
        out = add(out, b);
        return out;
    }

    vector<Value*> parameters() {
        vector<Value*> p = w;
        p.push_back(b);
        return p;
    }
};

struct Linear {
    vector<Neuron> neurons;
    bool use_relu;

    Linear(int nin, int nout, bool use_relu = true): use_relu(use_relu) {
        for(int i = 0; i < nout; i++) {
            neurons.emplace_back(nin);
        }
    }

    vector<Value*> operator()(const vector<Value*>&x) {
        vector<Value*> out;
        for(auto &n: neurons){
            out.push_back(use_relu ? relu(n(x)) : n(x));
        }
        return out;
    }

    vector<Value*> parameters() {
        vector<Value*> p;
        for(auto &n: neurons) {
            auto np = n.parameters();
            p.insert(p.end(), np.begin(), np.end());
        }
        return p;
    }
};

struct MLP {
    vector<Linear> layers;

    MLP(int nin, const vector<int>& nouts) {
        int in = nin;
        for(int i = 0; i < (int)nouts.size(); i++) {
            bool use_relu = (i != (int)nouts.size() - 1);
            layers.emplace_back(in, nouts[i], use_relu);
            in = nouts[i];
        }
    }

    vector<Value*> operator()(const vector<Value*> &x) {
        vector<Value*> out = x;
        for(auto &l: layers) out = l(out);
        return out;
    }

    vector<Value*> parameters() {
        vector<Value*> p;
        for(auto &l: layers) {
            auto lp = l.parameters();
            p.insert(p.end(), lp.begin(), lp.end());
        }
        return p;
    }
};