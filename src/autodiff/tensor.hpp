/*

Tensor class

It should support the following methods

- storing shape
- flat storage of Value*
- indexing
- reshape
- printing
- elementwise add
- elementwise multiply
- matrix multiply
- ReLU/tanh
- sum along all elements

*/

#pragma once
#include "value.hpp"

struct Tensor {
    vector<int> shape;
    vector<Value*> data;

    /*
    
    Constructor
    
    */ 

    // Empty Tensor
    Tensor(){}
   
    // Initializes a tensor with given shape 
    // and assigns Value type to every cell
    // with user given default value
    Tensor(const vector<int> &shape, double init_val = 0.0): shape(shape) {
        int total = 1;
        for(int d: shape) total *= d;
        data.assign(total, Value::create(init_val));
    }

    /*
    
    Method to get total number of elements

    */
    int numel() const {
        return (int)data.size();
    }

    int ndim() const {
        return (int)shape.size();
    }

    static Tensor zeros(const vector<int> &shape) {
        return Tensor(shape, 0.0);
    }

    static Tensor from_vector(const vector<double> &vals, const vector<int> &shape) {
        Tensor t;
        t.shape = shape;
        for(double x: vals) t.data.push_back(Value::create(x));
        return t;
    }

    int flatten_index(const vector<int> &idx) const {
        assert(idx.size() == shape.size());

        int flat = 0, stride = 1;
        for(int i = (int)shape.size() - 1; i >= 0; i--){
            assert(idx[i] >= 0 && idx[i] < shape[i]);
            flat += idx[i] * stride;
            stride *= shape[i];
        }
        return flat;
    }

    Value* at(const vector<int> &idx) const {
        return data[flatten_index(idx)];
    }

    Value*& at(const vector<int> &idx) {
        return data[flatten_index(idx)];
    }

    Tensor reshape(const vector<int> &new_shape) const {
        int old_total = 1, new_total = 1;
        for(int x: shape) old_total *= x;
        for(int x: new_shape) new_total *= x;
        
        assert(old_total == new_total);

        Tensor out;
        out.shape = new_shape;
        out.data = data;
        return out;
    }

    void print_shape() const {
        cout<<"(";
        for(int i = 0; i < (int)shape.size(); i++){
            cout<<shape[i];
            if(i+1 < (int)shape.size()) cout<<", ";
        }
        cout<<")\n";
    }

    void print_data() const {
        cout<<"[";
        for(int i = 0; i < (int)data.size(); i++){
            cout<<data[i]->data;
            if(i+1 < (int)data.size()) cout<<", ";
        }
        cout<<"]\n";
    }
};

// Elementwise addition
inline Tensor add(const Tensor&a, const Tensor&b) {
    assert(a.shape == b.shape);

    Tensor out;
    out.shape = a.shape;
    for(int i = 0; i < a.numel(); i++){
        out.data.push_back(add(a.data[i], b.data[i]));
    }
    return out;
}

// Elementwise multiplication
inline Tensor add(const Tensor&a, const Tensor&b) {
    assert(a.shape == b.shape);

    Tensor out;
    out.shape = a.shape;
    for(int i = 0; i < a.numel(); i++){
        out.data.push_back(mul(a.data[i], b.data[i]));
    }
    return out;
}

// ReLU
inline Tensor relu(const Tensor &a) {
    Tensor out;
    out.shape = a.shape;
    for(auto v: a.data) out.data.push_back(relu(v));
    return out;
}

// tanh
inline Tensor relu(const Tensor &a) {
    Tensor out;
    out.shape = a.shape;
    for(auto v: a.data) out.data.push_back(tanh_v(v));
    return out;
}

// sum all elements
inline Value* sum(const Tensor &a) {
    Value* out = Value::create(0.0);
    for(auto v: a.data) out = add(out, v);
    return out;
}

// matmul 
// a = [m, n], b = [n, p]
// out = [m, p]
inline Tensor matmul(const Tensor &a, const Tensor &b) {
    assert(a.ndim() == 2 && b.ndim() == 2);
    int m = a.shape[0], n = a.shape[1];
    int n_ = b.shape[0], p = b.shape[1];

    assert(n == n_);

    Tensor out = Tensor::zeros({m, p});
    
    for(int row = 0; row < m; row++){
        for(int col = 0; col < p; col++){
            Value* cell = Value::create(0.0);
            
            for(int k = 0; k < n; k++){
                cell = add(cell, mul(a.at({row, k}), b.at({k, col})));
            }
            out.at({row, col}) = cell;
        }
    }
}