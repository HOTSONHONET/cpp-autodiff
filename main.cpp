#include "nn.hpp";

using namespace std;

Value* softmax_cross_entropy(vector<Value*> logits, int target) {
    double maxx = -1e18;
    for(auto x: logits) maxx = max(maxx, x->data);

    vector<Value*> shifted;
    for(auto x: logits) shifted.push_back(sub(x, Value::create(maxx)));

    vector<Value*> exps;
    for(auto x: shifted) exps.push_back(exp_v(x));

    Value* sum_exp = Value::create(0.0);
    for(auto e: exps) sum_exp = add(sum_exp, e);

    vector<Value*> probs;
    for(auto e: exps) probs.push_back(divv(e, sum_exp));

    Value* logp = log_v(probs[target]);
    Value* loss = mul(Value::create(-1.0), logp);
    return loss;
}


void zero_grad(const vector<Value*> &params) {
    for(auto p: params) p->grad = 0.0;
}

void step_sgd(const vector<Value*> &params, double lr) {
    for(auto p: params) {
        p->data -= lr * p->grad;
    }
}