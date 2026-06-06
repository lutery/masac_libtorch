#pragma once

#include <torch/torch.h>
#include <deque>
#include <tuple>
#include <vector>
#include <cstdlib>
#include <algorithm>

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity);

    void add(torch::Tensor state,
             torch::Tensor action,
             torch::Tensor reward,
             torch::Tensor next_state,
             torch::Tensor done);

    size_t size() const;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    sample(size_t batch_size);

    std::tuple<torch::Tensor, torch::Tensor>
    sample2(size_t batch_size);


private:
    size_t capacity_; // 缓冲区的容量
    std::deque<torch::Tensor> states_, actions_, rewards_, next_states_, dones_;
};
