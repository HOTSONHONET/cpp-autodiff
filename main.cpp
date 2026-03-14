#include "nn.hpp"

using namespace std;
#define nline "\n"

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


int main() {
    vector<vector<double>> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    vector<int> Y = {0, 1, 1, 0};

    MLP model(2, {8, 2});
    auto params = model.parameters();

    for(int epoch = 0; epoch < 100; epoch++) {
        Value* total_loss = Value::create(0.0);

        for(int i = 0; i < (int) X.size(); i++) {
            vector<Value*> x = {
                Value::create(X[i][0]),
                Value::create(X[i][1]),
            };

            auto logits = model(x);
            Value* loss = softmax_cross_entropy(logits, Y[i]);
            total_loss = add(total_loss, loss);
        }

        zero_grad(params);
        total_loss->backward();
        step_sgd(params, 0.05);

        cout << "epoch " << epoch << " loss = " << total_loss->data << nline;
    }

    cout<<nline;
    cout<<"Predictions"<<nline;

    for(int i = 0; i < (int) X.size(); i++){
        vector<Value*> x = {
            Value::create(X[i][0]),
            Value::create(X[i][1]),
        };
        auto logits = model(x);
        int pred = logits[1]->data > logits[0]->data ? 1 : 0;
        cout<< X[i][0]<< " " << X[i][1] << " -> " << pred <<nline;
    }

    return 0;
}