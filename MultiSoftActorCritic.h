#pragma once
#include <torch/torch.h>
#include <vector>
#include <memory>
#include "Models.h"       // Reuse your Actor / Critic implementations
#include "ReplayBuffer.h" // Reuse your buffer (store reward as [1, N])

/**
 * Multi-Agent SAC with INDEPENDENT centralized Twin-Q per agent (CTDE):
 * - Each agent i has its own: Actor, temperature alpha_i, and Twin-Q_i (with targets).
 * - Critic inputs are centralized: concat(all local observations), concat(all actions).
 * - Actor i is updated against its OWN critic Q_i.
 * - Reward is per-agent: R[:, i].
 * - Done is shared here (shape [B,1]) to keep the example simple.
 */
class MultiAgentSAC
{
public:
    MultiAgentSAC(int numberOfAgents,
                  int localObservationDimension,
                  int actionDimension,
                  double initialAlpha,
                  double alphaLearningRate,
                  double discountFactor,
                  double targetUpdateTau,
                  double learningRate)
        : numberOfAgents_(numberOfAgents),
          localObservationDimension_(localObservationDimension),
          actionDimension_(actionDimension),
          globalStateDimension_(numberOfAgents * localObservationDimension),
          jointActionDimension_(numberOfAgents * actionDimension),
          gamma_(discountFactor),
          tau_(targetUpdateTau),
          targetEntropy_(-static_cast<double>(actionDimension)) // standard SAC heuristic
    {
        // Per-agent actor and temperature 这里是给智能体/优化器提前预约空间
        // 看起来使用vector来存储这些信息
        actors_.reserve(numberOfAgents_);
        actorOptimizers_.reserve(numberOfAgents_);
        logAlphas_.reserve(numberOfAgents_);
        alphaOptimizers_.reserve(numberOfAgents_);
        temperatures_.resize(numberOfAgents_);

        for (int i = 0; i < numberOfAgents_; ++i)
        {
            actors_.emplace_back(Actor(localObservationDimension_, actionDimension_));
            actors_.back()->to(torch::kF64); // 设置浮点数
            actorOptimizers_.emplace_back(actors_.back()->parameters(), learningRate); // 这里应该是一种c++对象构造的方法，直接传入参数隐式构造优化器

            // todo 这个应该是可学习的参数，具体用在哪里需要排查洗
            auto logAlpha = torch::log(torch::tensor({initialAlpha}, torch::kF64)).set_requires_grad(true);
            logAlphas_.push_back(logAlpha);
            alphaOptimizers_.emplace_back(std::vector<torch::Tensor>{logAlphas_.back()}, alphaLearningRate);
            temperatures_[i] = initialAlpha; // todo 这个用在哪里
        }

        // Per-agent twin critics (centralized inputs) 看起来有多少个环境智能体就有多少个评价网络
        // 每个动作预测器对应两个Q值评价器
        critics1_.reserve(numberOfAgents_);
        critics2_.reserve(numberOfAgents_);
        targetCritics1_.reserve(numberOfAgents_);
        targetCritics2_.reserve(numberOfAgents_);
        critic1Optimizers_.reserve(numberOfAgents_);
        critic2Optimizers_.reserve(numberOfAgents_);

        for (int i = 0; i < numberOfAgents_; ++i)
        {
            critics1_.emplace_back(Critic(globalStateDimension_, jointActionDimension_)); // 评价网络是全局观察
            critics2_.emplace_back(Critic(globalStateDimension_, jointActionDimension_));
            targetCritics1_.emplace_back(Critic(globalStateDimension_, jointActionDimension_));
            targetCritics2_.emplace_back(Critic(globalStateDimension_, jointActionDimension_));

            critics1_.back()->to(torch::kF64);
            critics2_.back()->to(torch::kF64);
            targetCritics1_.back()->to(torch::kF64);
            targetCritics2_.back()->to(torch::kF64);

            hardCopy(critics1_.back(), targetCritics1_.back());
            hardCopy(critics2_.back(), targetCritics2_.back());

            critic1Optimizers_.emplace_back(std::make_unique<torch::optim::Adam>(critics1_.back()->parameters(), learningRate));
            critic2Optimizers_.emplace_back(std::make_unique<torch::optim::Adam>(critics2_.back()->parameters(), learningRate));
        }
    }

    // Forward a batch global state to extract the local slice for agent i
    // 这里是从全局状态中切片出每个智能体的局部观察，输入是一个批次的全局状态，输出是对应智能体的局部观察
    torch::Tensor sliceLocalObservation(const torch::Tensor &batchGlobalState, int agentIndex) const
    {
        const int start = agentIndex * localObservationDimension_;
        // 这是 LibTorch 的张量切片操作，等价于 Python 的 tensor[:, start:end]。
        '''
        这是 LibTorch 的张量切片操作，等价于 Python 的 `tensor[:, start:end]`。

        ```cpp
        // batchGlobalState 形状: [B, globalStateDimension_]  即 [batch_size, 12]
        // agentIndex = 0 时: start=0,  取列 [0:4)   → 智能体0 的局部观测
        // agentIndex = 1 时: start=4,  取列 [4:8)   → 智能体1 的局部观测
        // agentIndex = 2 时: start=8,  取列 [8:12)  → 智能体2 的局部观测

        return batchGlobalState.index({
            torch::indexing::Slice(),                                   // 维度0: 取所有行 (B)
            torch::indexing::Slice(start, start + localObservationDimension_) // 维度1: 取列 [start, start+4)
        });
        ```

        两个参数对应两个维度：
        - 第一个 `Slice()`（无参数）= `:`，保留所有 batch
        - 第二个 `Slice(start, end)` = `start:end`，切出 `[pos_x, pos_y, goal_x, goal_y]` 四列

        全局状态是 3 个智能体观测的拼接 `[a0的4维 | a1的4维 | a2的4维]`，这个函数按智能体索引把对应片段切出来，返回形状 `[B, 4]` 的张量。
        '''
        return batchGlobalState.index({torch::indexing::Slice(),
                                       torch::indexing::Slice(start, start + localObservationDimension_)});
    }

    // Concatenate along feature dimension
    static torch::Tensor concatAlongFeature(const std::vector<torch::Tensor> &parts)
    {
        return torch::cat(parts, /*dim=*/1);
    }

    // Stochastic action sampling for a single-step (batch size = 1 per agent input)
    torch::Tensor act(const std::vector<torch::Tensor> &localObservationsBatch1)
    {
        std::vector<torch::Tensor> actionParts; // 存储每个智能体预测的动作
        actionParts.reserve(numberOfAgents_); // 这里是给每一个智能体的动作提前预约空间
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            auto [action_i, logProb_i] = actors_[i]->sample(localObservationsBatch1[i]); // 这里是给每一个智能体单独采样动作，返回每一个
            /**
             * 这是**抑制"未使用变量"警告**的惯用写法。
```cpp
auto [action_i, logProb_i] = actors_[i]->sample(localObservationsBatch1[i]);
(void)logProb_i; // 告诉编译器：我知道这个变量没用，别警告
```

`actor->sample()` 返回 `{action, log_prob}` 两个值，但这里只需要 action（推理阶段用不到 log_prob）。如果用 `std::ignore` 或 `[[maybe_unused]]`，在不同编译器/标准下可能不一致，而 `(void)expr` 是零开销、跨所有编译器都生效的写法。
             */
            (void)logProb_i; // not needed in inference
            actionParts.push_back(action_i);
        }
        return concatAlongFeature(actionParts); // shape [1, jointActionDimension_]
    }

    // One training step using a minibatch sampled from the buffer
    // Buffer sample shapes:  B is batch size
    //   GlobeState:     [B, globalStateDimension_]
    //   Action:         [B, jointActionDimension_]
    //   Reward:         [B, numberOfAgents_]
    //   GlobeStateNext: [B, globalStateDimension_]
    //   Done:           [B, 1]
    void trainStep(ReplayBuffer &buffer, size_t batchSize)
    {
        if (buffer.size() < batchSize)
            return;

        auto [batchState, batchAction, batchReward, batchNextState, batchDone] = buffer.sample(batchSize);
        // 输出各变量的 shape
        // std::cout << "batchState shape: " << batchState.sizes() << std::endl;
        // std::cout << "batchAction shape: " << batchAction.sizes() << std::endl;
        // std::cout << "batchReward shape: " << batchReward.sizes() << std::endl;
        // std::cout << "batchNextState shape: " << batchNextState.sizes() << std::endl;
        // std::cout << "batchDone shape: " << batchDone.sizes() << std::endl;
        // Next joint action sampled from current actors (for all agents)
        // 存储对每个智能体的下一个动作和对应的log概率，后续会用来计算目标Q值和更新actor
        std::vector<torch::Tensor> nextActionParts, nextLogProbParts; // nextActionParts[i].sizes() = [B, actionDimension_]
        nextActionParts.reserve(numberOfAgents_);
        nextLogProbParts.reserve(numberOfAgents_);
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            auto nextLocalObs_i = sliceLocalObservation(batchNextState, i);
            auto [nextAction_i, nextLogProb_i] = actors_[i]->sample(nextLocalObs_i); // 根据切出来的下一个状态，预测下一个动作和对应的log概率
            nextActionParts.push_back(nextAction_i);
            nextLogProbParts.push_back(nextLogProb_i);
        }
        auto batchNextAction = concatAlongFeature(nextActionParts); // 拼接成完整的联合动作 [B, jointActionDimension_]

        // Sum of next log-probabilities across agents (appears in target)
        // 构造 SAC critic target 中的多智能体熵项
        // 标准单智能体 SAC 的 target： y = r + γ·(1-done)·(min(Q'₁, Q'₂) - α · log_prob)
        // 多智能体 CTDE 中，每个智能体 i 的 critic 在估计未来价值时，需要考虑所有智能体策略的随机性（熵），但用各自的温度 αⱼ 加权： y_i = r_i + γ·(1-done)·(min(Q'₁_i, Q'₂_i) - Σⱼ αⱼ · log_probⱼ)
        auto sumNextLogProb = torch::zeros_like(nextLogProbParts[0]);
        // for (auto &lp : nextLogProbParts)
        //     sumNextLogProb = sumNextLogProb + lp;
        // 这里的值通常是负数

        for (int i = 0; i < numberOfAgents_; ++i)
        {
            sumNextLogProb = sumNextLogProb + nextLogProbParts[i] * temperatures_[i];
        }

        // ---- Per-agent critic updates (independent centralized Q for each agent) ----
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            torch::Tensor targetQ_i;
            {
                torch::NoGradGuard noGrad;

                auto q1Next = targetCritics1_[i]->forward(batchNextState, batchNextAction); // [B,1]
                auto q2Next = targetCritics2_[i]->forward(batchNextState, batchNextAction); // [B,1]
                auto minQNext = torch::min(q1Next, q2Next);

                // R[:, i] -> shape [B, 1]
                auto rewardColumn_i = batchReward.index({torch::indexing::Slice(), i}).view({-1, 1});
                // auto rewardSum = batchReward.sum(1, /*keepdim=*/true);
                // 由于sumNextLogProb通常是负数，并且越不确定越负，所以这里的可以增加探索
                // 比如错误的动作被施加了更大的Q值，后续可能会倾向选择这个动作，但是由于这个动作是错误的
                // 在探索后期Q值会减少，从而又其他了其他动作的Q值
                targetQ_i = rewardColumn_i + gamma_ * (1.0 - batchDone) * (minQNext - sumNextLogProb);
            }

            auto q1 = critics1_[i]->forward(batchState, batchAction);
            auto q2 = critics2_[i]->forward(batchState, batchAction);
            auto criticLoss = torch::mse_loss(q1, targetQ_i) + torch::mse_loss(q2, targetQ_i);

            critic1Optimizers_[i]->zero_grad();
            critic2Optimizers_[i]->zero_grad();
            criticLoss.backward();
            critic1Optimizers_[i]->step();
            critic2Optimizers_[i]->step();
        }

        // ---- Per-agent actor and temperature updates ----
        // Re-sample current joint actions A_pi for policy evaluation
        // currentActionParts：存储每个Agent选择的动作
        // currentLogProbParts： 存储每个Agent选择对应动作的概率密度
        std::vector<torch::Tensor> currentActionParts(numberOfAgents_), currentLogProbParts(numberOfAgents_);
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            auto localObs_i = sliceLocalObservation(batchState, i); // 从全局状态中切分每个Agent的状态
            auto [action_i, logProb_i] = actors_[i]->sample(localObs_i); // 根据局部的Agent状态采样动作和动作的概率密度的log值
            currentActionParts[i] = action_i;
            currentLogProbParts[i] = logProb_i;
        }
        // 对每个agent进行独立的actor更新
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            // 构建联合动作：当前agent保持梯度，其他agent断开梯度
            std::vector<torch::Tensor> jointActionParts(numberOfAgents_);
            for (int j = 0; j < numberOfAgents_; ++j)
            {
                // 这里是构建联合动作的关键部分，智能体 i 的动作保持梯度，其他智能体的动作使用 detach() 来阻断梯度流，这样在更新智能体 i 的 actor 时，只会影响智能体 i 的策略，而不会影响其他智能体的策略。
                jointActionParts[j] = (j == i) ? currentActionParts[j] : currentActionParts[j].detach();
            }
            // 将所有动作组合在一起
            auto jointAction = concatAlongFeature(jointActionParts);

            // 使用智能体 i 的 critic 计算 Q 值
            auto qMin_i = torch::min(
                critics1_[i]->forward(batchState, jointAction),
                critics2_[i]->forward(batchState, jointAction));
            // - qMin_i 代表如果想要将值越小，则qMin_i要变大，而要变化那么所预测的动作就要能够使得Q值变化，从而使得预测的动作能够靠近Q值大的动作，如果当前选择的动作Q值时正，那么就会加大这个动作的概率；如果是负数，为了减小，那么就会使得预测的动作远离选择动作，从而使得概率减少
            // temperatures_[i] * currentLogProbParts[i]：表示动作的概率密度，如果选择某个动作越确定，会导致概率密度越大，而损失又会将概率密度缩小，从而降低某个动作被选的概率
            auto actorLoss_i = (temperatures_[i] * currentLogProbParts[i] - qMin_i).mean();
            // update actor i 根据以上的规则，从而更新动作预测
            actorOptimizers_[i].zero_grad();
            actorLoss_i.backward();
            actorOptimizers_[i].step();

            // Temperature (alpha) update
            // α 必须是正数（温度不能为负）。如果直接优化 α，梯度更新可能把它推出正数范围。优化 logα 就没有这个限制——logα 可以是任意实数，然后 α = exp(logα) 自动保证正数。
            // targetEntropy_是标准的目标熵，由于熵一般是负数，所以这里采用加号
            // 如果策略过于确定，那么currentLogProbParts作为选择动作的概率也会偏大，导致currentLogProbParts[i].detach() + targetEntropy_为正数，今儿导致
            // -(logAlphas_[i] * (currentLogProbParts[i].detach() + targetEntropy_)) 整体为负数，为了梯度下降，那么logAlphas_就会增大
            // logAlphas_ 增加就会加大 熵在之前训练时的影响权重，从而将选择的动作概率下降
            // 反之策略不确定，那么currentLogProbParts也会偏小，导致currentLogProbParts[i].detach() + targetEntropy_为负数，进而导致
            // -(logAlphas_[i] * (currentLogProbParts[i].detach() + targetEntropy_)) 整体为正数，为了梯度下降，那么logAlphas_就会减小
            // logAlphas_ 减小就会减少 熵在之前训练时的影响权重，从而将选择的动作概率上升
            auto alphaLoss_i = -(logAlphas_[i] * (currentLogProbParts[i].detach() + targetEntropy_)).mean();
            alphaOptimizers_[i].zero_grad();
            alphaLoss_i.backward();
            alphaOptimizers_[i].step();
            temperatures_[i] = logAlphas_[i].exp().item<double>();
        }

        // ---- Soft update target critics ----
        torch::NoGradGuard noGrad;
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            softUpdate(critics1_[i], targetCritics1_[i]);
            softUpdate(critics2_[i], targetCritics2_[i]);
        }
    }

private:
    int numberOfAgents_; // 智能体的个数
    int localObservationDimension_; // 观察空间的维度
    int actionDimension_; // 动作空间的维度
    int globalStateDimension_; // 全局观察空间的维度
    int jointActionDimension_; // 全局动作空间的维度
    
    double gamma_; // 这个应该是计算bellman Q值的比例
    double tau_;    // 这几个应该是将参数同步到目标网络时的权重
    double targetEntropy_; // 目标动作熵

    // Per-agent policies and temperatures
    std::vector<Actor> actors_; // 动作预测智能体，每个仅观察局部obs和动作
    std::vector<torch::optim::Adam> actorOptimizers_;

    std::vector<torch::Tensor> logAlphas_; // 每个智能体对应的alhpas参数
    std::vector<double> temperatures_;
    std::vector<torch::optim::Adam> alphaOptimizers_;

    // Per-agent centralized twin critics + targets
    std::vector<Critic> critics1_, critics2_, targetCritics1_, targetCritics2_;
    std::vector<std::unique_ptr<torch::optim::Adam>> critic1Optimizers_, critic2Optimizers_;

    /**
     * Soft update: target ← (1−τ)·target + τ·source
     */
    void softUpdate(const Critic &source, Critic &target)
    {
        auto targetParams = target->parameters();
        auto sourceParams = source->parameters();
        for (size_t p = 0; p < targetParams.size(); ++p)
        {
            targetParams[p].data().mul_(1.0 - tau_);
            targetParams[p].data().add_(tau_ * sourceParams[p].data());
        }
    }
    /**
     * Hard update: target ← source
     */
    void hardCopy(const Critic &source, Critic &target)
    {
        auto targetParams = target->parameters();
        auto sourceParams = source->parameters();
        for (size_t p = 0; p < targetParams.size(); ++p)
        {
            targetParams[p].data().copy_(sourceParams[p].data());
        }
    }
};
