#pragma once
#include "value.hpp"

struct Neuron {
    vector<Value*> w;
    Value* b;

    Neuron(int nin) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
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

    Linear(int nin, int nout) {
        for(int i = 0; i < nout; i++) {
            neurons.emplace_back(nin);
        }
    }

    vector<Value*> operator()(const vector<Value*>&x) {
        vector<Value*> out;
        for(auto &n: neurons) out.push_back(n(x));
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
        for(int nout: nouts) {
            layers.emplace_back(in, nout);
            in = nout;
        }
    }

    vector<Value*> operator()(const vector<Value*> &x) {
        vector<Value*> out = x;
        for(int i = 0; i < (int) layers.size(); i++) {
            out = layers[i](out);
        }
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