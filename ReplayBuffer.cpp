#include "ReplayBuffer.h"

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity_(capacity) {}

void ReplayBuffer::add(torch::Tensor state,
                       torch::Tensor action,
                       torch::Tensor reward,
                       torch::Tensor next_state,
                       torch::Tensor done) {
    if (states_.size() >= capacity_) {
        states_.pop_front();
        actions_.pop_front();
        rewards_.pop_front();
        next_states_.pop_front();
        dones_.pop_front();
    }

    states_.push_back(state.detach().clone().view({-1}));
    actions_.push_back(action.detach().clone().view({-1}));
    rewards_.push_back(reward.detach().clone().view({-1}));
    next_states_.push_back(next_state.detach().clone().view({-1}));
    dones_.push_back(done.detach().clone().view({-1}));
}

size_t ReplayBuffer::size() const {
    return states_.size();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,torch::Tensor, torch::Tensor>
ReplayBuffer::sample(size_t batch_size)
 {
    std::vector<torch::Tensor> state_batch;
    std::vector<torch::Tensor> action_batch;
    std::vector<torch::Tensor> reward_batch;
    std::vector<torch::Tensor> next_state_batch;
    std::vector<torch::Tensor> done_batch;

    for (size_t i = 0; i < batch_size; ++i) {
        int index = rand() % states_.size();  // random index 看来本算法不要求序列化采样，直接随机采样即可

        state_batch.push_back(states_[index]);
        action_batch.push_back(actions_[index]);
        reward_batch.push_back(rewards_[index]);
        next_state_batch.push_back(next_states_[index]);
        done_batch.push_back(dones_[index]);
    }

    torch::Tensor states = torch::stack(state_batch);
    torch::Tensor actions = torch::stack(action_batch);
    torch::Tensor rewards = torch::stack(reward_batch);
    torch::Tensor next_states = torch::stack(next_state_batch);
    torch::Tensor dones = torch::stack(done_batch);

    return std::make_tuple(states, actions, rewards, next_states, dones);
}
