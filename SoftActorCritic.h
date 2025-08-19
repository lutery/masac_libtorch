#pragma once
#include <torch/torch.h>
#include "Models.h"       // Requires SACActor / SACCritic definitions
#include "ReplayBuffer.h" // Requires ReplayBuffer definition

class SAC {
public:
    SAC(int state_dim,
        int action_dim,
        double alpha,
        double alpha_lr,
        double gamma,
        double tau,
        double lr)
        : actor(state_dim, action_dim),
        critic1(state_dim , action_dim),
        critic2(state_dim , action_dim),
        target1(state_dim , action_dim),
        target2(state_dim , action_dim),
        log_alpha(torch::log(torch::tensor({alpha}, torch::kF64)).set_requires_grad(true)), // ✅ 初始化 log_alpha
        alpha_opt({log_alpha}, alpha_lr), 
        actor_opt(actor->parameters(), lr),
        critic1_opt(critic1->parameters(), lr),
        critic2_opt(critic2->parameters(), lr),
        alpha_(alpha),
        gamma_(gamma),
        tau_(tau),
        target_entropy(-static_cast<double>(action_dim))
    {
        hard_update(critic1, target1);
        hard_update(critic2, target2);

        actor  ->to(torch::kF64);
        critic1->to(torch::kF64);
        critic2->to(torch::kF64);
        target1->to(torch::kF64);
        target2->to(torch::kF64);
    }

    /**
     * Performs one training step using a mini-batch from the replay buffer.
     * If there are not enough samples, the function returns without doing anything.
     */
    void train_step(ReplayBuffer& buffer, size_t batch_size)
    {
        if (buffer.size() < batch_size) return;

        auto [s_batch, a_batch, r_batch, ns_batch, d_batch] = buffer.sample(batch_size);
        //---------- Critic update ----------
        torch::Tensor q_target;
        {
            torch::NoGradGuard no_grad;

            auto [a_next, logp_next] = actor->sample(ns_batch);
            auto q1_next = target1->forward(ns_batch, a_next);
            auto q2_next = target2->forward(ns_batch, a_next);
            q_target = r_batch +
                    gamma_ * (1 - d_batch) *
                    (torch::min(q1_next, q2_next) - alpha_ * logp_next);
        }       
        auto q1 = critic1->forward(s_batch, a_batch);
        auto q2 = critic2->forward(s_batch, a_batch);
        auto critic_loss = torch::mse_loss(q1, q_target) + torch::mse_loss(q2, q_target);

        critic1_opt.zero_grad();
        critic2_opt.zero_grad();
        critic_loss.backward();
        critic1_opt.step();
        critic2_opt.step();

        //---------- Actor update ----------
        auto [a_pred, logp_pred] = actor->sample(s_batch);
        auto q_pred = torch::min(critic1->forward(s_batch, a_pred),
                                 critic2->forward(s_batch, a_pred));
        auto actor_loss = (alpha_ * logp_pred - q_pred).mean();

        actor_opt.zero_grad();
        actor_loss.backward();
        actor_opt.step();

        //---------- Alpha update ----------
        // auto alpha_loss = -(log_alpha * (logp_pred + target_entropy).detach()).mean();
        auto alpha_loss = -(log_alpha * (logp_pred.detach() + target_entropy)).mean();
        alpha_opt.zero_grad();
        alpha_loss.backward();
        alpha_opt.step();
        alpha_ = log_alpha.exp().item<double>();  // update scalar        

        //---------- Soft update target networks ----------
        torch::NoGradGuard no_grad;
        soft_update(critic1, target1);
        soft_update(critic2, target2);
    }

    // Optionally: save/load model weights
    void save(const std::string& path_prefix) const {
        torch::save(actor,   path_prefix + "_actor.pt");
        torch::save(critic1, path_prefix + "_critic1.pt");
        torch::save(critic2, path_prefix + "_critic2.pt");
        torch::save(log_alpha, path_prefix + "_log_alpha.pt");
        std::cout<< "best model saved "<<std::endl;
    }

    void load(const std::string& path_prefix) {
        torch::load(actor,   path_prefix + "_actor.pt");
        torch::load(critic1, path_prefix + "_critic1.pt");
        torch::load(critic2, path_prefix + "_critic2.pt");
        torch::load(log_alpha, path_prefix + "_log_alpha.pt");
        hard_update(critic1, target1);
        hard_update(critic2, target2);
        alpha_ = log_alpha.exp().item<double>();
        std::cout<< "best model loaded "<<std::endl;
    }


    // Public members for inference or external access
    Actor   actor;
    Critic  critic1, critic2;   // Main Q-networks
    Critic  target1, target2;   // Target Q-networks

private:
    torch::Tensor log_alpha;
    torch::optim::Adam actor_opt;
    torch::optim::Adam critic1_opt;
    torch::optim::Adam critic2_opt;
    torch::optim::Adam alpha_opt;
    
    double alpha_, gamma_, tau_, target_entropy;

    /**
     * Soft update: target ← (1−τ)·target + τ·source
     */
    void soft_update(const Critic& source, Critic& target) {
        torch::NoGradGuard no_grad;
        auto target_params = target->parameters();
        auto source_params = source->parameters();
        for (size_t i = 0; i < target_params.size(); ++i) {
            target_params[i].data().mul_(1.0 - tau_);
            target_params[i].data().add_(tau_ * source_params[i].data());
        }
    }

    /**
     * Hard update: target ← source
     */
    void hard_update(const Critic& source, Critic& target) {
        torch::NoGradGuard no_grad;
        auto target_params = target->parameters();
        auto source_params = source->parameters();
        for (size_t i = 0; i < target_params.size(); ++i) {
            target_params[i].data().copy_(source_params[i].data());
        }
    }

};
