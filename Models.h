#pragma once

#include <torch/torch.h>
#include <math.h>
// ========== Actor Network========= //
struct ActorImpl : public torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, mean{nullptr}, log_std{nullptr};
    ActorImpl(int64_t input_dim, int64_t output_dim) { // 输入的一个是局部观察的维度，一个是局部智能体的动作维度
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 16));
        fc2 = register_module("fc2", torch::nn::Linear(16, 32));
        mean = register_module("mean", torch::nn::Linear(32, output_dim));
        log_std = register_module("log_std", torch::nn::Linear(32, output_dim));
    }
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        auto mu = mean->forward(x);
        auto log_std_out = torch::clamp(log_std->forward(x), -20, 2); // numerical stability
        return std::make_tuple(mu, log_std_out);
    }

    std::tuple<torch::Tensor, torch::Tensor> sample(torch::Tensor x) {
        // a = tanh(\miu+\sigma*\epsilon)
        auto [mu, log_std] = forward(x);
        auto std = torch::exp(log_std);
        auto eps = torch::randn_like(std);
        auto action_raw = mu + std * eps;
        auto action = action_raw.tanh();
        // \log \pi(\mathbf{a}|s) = \log \pi_{\text{raw}}(\mathbf{u}|s) - \sum_{i} \log \left( 1 - \tanh^2(u_i) + \epsilon \right)
        auto log_prob = (-0.5 * (eps.pow(2) + 2*log_std + std::log(2*M_PI)) - 
                        torch::log(1 - action.pow(2) + 1e-6));  // 防止 log(0)
        log_prob = log_prob.sum(1, true);
        return {action, log_prob};
    }
};
TORCH_MODULE(Actor); // TORCH_MODULE(Actor) 是 LibTorch 提供的一个宏，它在编译期自动生成以下代码：

// ========== Critic Q-Network ========== //
struct CriticImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    CriticImpl(int64_t input_dim, int64_t action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim + action_dim, 32));
        fc2 = register_module("fc2", torch::nn::Linear(32, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 1));
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = torch::cat({state, action}, 1);
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

TORCH_MODULE(Critic);