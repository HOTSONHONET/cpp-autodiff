#pragma once
#include <bits/stdc++.h>

using namespace std;

struct Value {
    double data;
    double grad = 0.0;

    vector<Value*> prev;
    function<void()> _backward;

    string op;

    Value(double data) : data(data), _backward([](){}) {}

    static Value* create(double x){
        return new Value(x);
    }

    void backward() {
        vector<Value*> topo;
        unordered_set<Value*> visited;

        function<void(Value*)> build = [&](Value*v) {
            if(visited.count(v)) return;
            visited.insert(v);

            for(auto p: v->prev) build(p);
            topo.push_back(v);
        };

        build(this);
        this->grad = 1.0;

        reverse(topo.begin(), topo.end());
        for(auto v: topo) v->_backward();
    }
};

// ---- operations ----

inline Value* add(Value *a, Value *b) {
    Value *out = Value::create(a->data + b->data);

    out->prev = {a, b};
    out->op = "+";

    out->_backward = [a, b, out]() {
        a->grad += 1.0 * out->grad;
        b->grad += 1.0 * out->grad;
    };

    return out;
}

inline Value* sub(Value *a, Value *b) {
    Value *out = Value::create(a->data - b->data);

    out->prev = {a, b};
    out->op = "-";

    out->_backward = [a, b, out]() {
        a->grad += 1.0 * out->grad;
        b->grad += -1.0 * out->grad;
    };

    return out;
}

inline Value* mul(Value *a, Value *b) {
    Value* out = Value::create(a->data * b->data);
    out->prev = {a, b};
    out->op = "*";

    out->_backward = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };
    return out;
}

inline Value* divv(Value *a, Value *b){
    Value* out = Value::create(a->data / b->data);
    out->prev = {a, b};
    out->op = "/";

    out->_backward = [a, b, out]() {
        a->grad += (1.0 / b->data) * out->grad;
        b->grad += (-a->data / (b->data * b->data)) * out->grad;
    };
    return out;
}

inline Value* tanh_v(Value *a) {
    double t = tanh(a->data);
    Value* out = Value::create(t);
    out->prev = {a};
    out->op = "tanh";

    out->_backward = [a, out, t]() {
        a->grad += (1.0 - t * t) * out->grad;
    };
    return out;
}

inline Value* relu(Value *a) {
    double r = a->data > 0 ? a->data : 0.0;
    Value* out = Value::create(r);
    out->prev = {a};
    out->op = "relu"; 

    out->_backward = [a, out]() {
        a->grad += (a->data > 0 ? 1.0 : 0.0) * out->grad;
    };
    return out;
}

inline Value* exp_v(Value *a) {
    double e = std::exp(a->data);
    Value* out = Value::create(e);
    out->prev = {a};
    out->op = "exp";

    out->_backward = [a, out]() {
        a->grad += out->data * out->grad;
    };
    return out;
}

inline Value* log_v(Value *a){
    Value* out = Value::create(std::log(a->data));
    out->prev = {a};
    out->op = "log";

    out->_backward = [a, out]() {
        a->grad += (1.0 / a->data) * out->grad;
    };
    return out;
}
