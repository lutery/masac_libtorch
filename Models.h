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
        auto mu = mean->forward(x); // 预测动作的均值
        // 预测动作的标准差的对数，限制在[-20, 2]范围内，防止数值问题 为什么预测的是标准差的对数？因为标准差必须是正数，预测对数可以确保输出的标准差是正数，同时也有利于数值稳定性。为什么限制在[-20, 2]范围内？因为过大的标准差会导致动作分布过于分散，过小的标准差会导致动作分布过于集中，这两种情况都会影响训练的稳定性和效果。
        auto log_std_out = torch::clamp(log_std->forward(x), -20, 2); // numerical stability 限制log_std的范围，防止std过大或过小导致数值问题
        return std::make_tuple(mu, log_std_out);
    }

    std::tuple<torch::Tensor, torch::Tensor> sample(torch::Tensor x) {
        // a = tanh(\miu+\sigma*\epsilon)
        auto [mu, log_std] = forward(x); // 预测动作的均值和标准差的对数
        auto std = torch::exp(log_std); // 得到标准差
        auto eps = torch::randn_like(std); // 采样一个与标准差形状相同的标准正态分布噪声
        auto action_raw = mu + std * eps; // 采样得到原始动作（未经过tanh变换）
        auto action = action_raw.tanh(); // 经过tanh变换，限制动作在[-1, 1]范围内
        // \log \pi(\mathbf{a}|s) = \log \pi_{\text{raw}}(\mathbf{u}|s) - \sum_{i} \log \left( 1 - \tanh^2(u_i) + \epsilon \right)
        auto log_prob = (-0.5 * (eps.pow(2) + 2*log_std + std::log(2*M_PI)) -  // 高斯分布 log 密度
                        torch::log(1 - action.pow(2) + 1e-6));  // 防止 log(0)  tanh 修正项（≥ 0，推负方向）
        log_prob = log_prob.sum(1, true); // 将每个动作维度的 log_prob 加总，得到一个标量 log_prob，形状 [B, 1]，通常是负数
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