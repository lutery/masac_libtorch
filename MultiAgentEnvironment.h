#pragma once
#include <Eigen/Core>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <cstring>
#include <cmath>

enum MAStatus
{
    MA_PLAYING = 0,
    MA_DONE = 1,
    MA_RESETTING = 2
};

/**
 * A simple cooperative grid environment without physical interactions.
 * - Each agent i has a position and a goal: (pos_x, pos_y) -> (goal_x, goal_y).
 * - Each step applies a bounded displacement proportional to the action.
 * - Rewards are PER-AGENT (vector): distance reduction bonus + terminal bonus/penalty.
 * - Done is shared: episode ends if all agents reach their goals OR any agent goes out-of-bounds.
 * 
 * 核心任务：3 个智能体在 2D 平面上移动，各自到达自己的目标点。所有智能体到达目标才算成功（团队胜利），任一智能体越界（距离原点 >10）则团队失败。
 */
class MultiAgentEnvironment
{
public:
    explicit MultiAgentEnvironment(int numberOfAgents,
                                   const std::vector<Eigen::Vector2d> &initialGoals)
        : numberOfAgents_(numberOfAgents),
          positions_(numberOfAgents),
          goals_(initialGoals),
          reached_(numberOfAgents_, 0)
    {
        for (int i = 0; i < numberOfAgents_; ++i)
            positions_[i].setZero();
    }

private:
    int numberOfAgents_; // 智能体的个数
    int statedim_ = 4; // [pos_x, pos_y, goal_x, goal_y]
    std::vector<uint8_t> reached_; double reach_radius_=0.6, leave_radius_=0.8; // todo
public:
    std::vector<Eigen::Vector2d> positions_; // todo 这个看起来是智能体的位置
    std::vector<Eigen::Vector2d> goals_; // todo 这个看起来是智能体的要达到的目标

    torch::Tensor getLocalObservation(int agentIndex)
    {
        torch::Tensor obs = torch::zeros({1, 4}, torch::kF64);
        obs[0][0] = positions_[agentIndex][0];
        obs[0][1] = positions_[agentIndex][1];
        obs[0][2] = goals_[agentIndex][0];
        obs[0][3] = goals_[agentIndex][1];
        return obs;
    }

    // Global state: concatenation of all agents' local observations in fixed order
    torch::Tensor getGlobalState()
    {
        torch::Tensor state = torch::zeros({1, statedim_ * numberOfAgents_}, torch::kF64);
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            state[0][statedim_ * i + 0] = positions_[i][0];
            state[0][statedim_ * i + 1] = positions_[i][1];
            state[0][statedim_ * i + 2] = goals_[i][0];
            state[0][statedim_ * i + 3] = goals_[i][1];
        }
        return state;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int>
    Step(const torch::Tensor &jointAction)
    {
        // ===== 1) Record old distances =====
        std::vector<double> old_dist_(numberOfAgents_, 0.0);
        for (int i = 0; i < numberOfAgents_; ++i)
            old_dist_[i] = distance(positions_[i], goals_[i]);

        // ===== 2) Apply actions =====
        const double maxStepLength = 0.1;
        auto action = jointAction.squeeze().to(torch::kF64);
        std::vector<double> act_energy_i(numberOfAgents_, 0.0);
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            const double ax = action[2 * i].item<double>();
            const double ay = action[2 * i + 1].item<double>();
            act_energy_i[i] = ax * ax + ay * ay; // used for reward
            positions_[i][0] += maxStepLength * ax;
            positions_[i][1] += maxStepLength * ay;
        }

        // ===== 3) Calculate current distances and team statistics =====
        std::vector<double> distNow(numberOfAgents_, 0.0);
        double sum_d = 0.0;
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            distNow[i] = distance(positions_[i], goals_[i]);
            sum_d += distNow[i];
        }
        const double mean_d = sum_d / std::max(1, numberOfAgents_);
        double var_d = 0.0;
        for (int i = 0; i < numberOfAgents_; ++i)
        {
            const double t = distNow[i] - mean_d;
            var_d += t * t;
        }
        var_d /= std::max(1, numberOfAgents_);

        
        // ===== 4) Per-agent rewards (one-time goal bonus + no accumulation after reaching) =====
        torch::Tensor rewardVector = torch::zeros({1, numberOfAgents_}, torch::kF64);

        // Adjustable weights (fine-tune or set to zero to disable team terms)
        const double w_ind = 10.0;  // Individual distance reduction
        const double w_goal = 20.0; // One-time goal reward
        const double w_oob = 20.0;  // Out-of-bounds penalty
        const double w_act = 0.1;  // Action energy penalty (shared)

        const double collision_dmin = 0.5; // Collision threshold

        int reachedCount = 0;
        int outOfBoundsCount = 0;

        for (int i = 0; i < numberOfAgents_; ++i)
        {
            const double d_old = old_dist_[i];
            const double d_now = distNow[i];

            const bool was_reached = (reached_[i] != 0);
            const bool now_reached = (d_now < reach_radius_);

            // Hysteresis: after reaching, only consider "still reached" if not far from leave_radius_
            if (!was_reached && now_reached)
            {
                reached_[i] = 1; // First time reaching
            }
            else if (was_reached && d_now > leave_radius_)
            {
                reached_[i] = 0; // Clearly left the target area
            }

            double r_i = 0.0;

            // if (!was_reached)
            if(true)
            {
                const double delta = d_old - d_now; // Positive means closer
                r_i += w_ind * (d_old - d_now);
            }

            // One-time goal reward: only given when first entering reach_radius_
            if (!was_reached && now_reached)
            {
                r_i += w_goal;
            }

            // Out-of-bounds penalty
            if (d_now > 10.0)
            {
                r_i -= w_oob;
                outOfBoundsCount++;
            }

            
            if (!reached_[i]) {
                const double delta = d_old - d_now; // Positive means closer
                if (delta < 0) r_i += -1.0;
            }

            if (reached_[i])
                reachedCount++;

            rewardVector[0][i] = r_i;
        }

        // ===== 5) Shared termination conditions =====
        const bool all_reached = (reachedCount == numberOfAgents_);
        const bool any_oob = (outOfBoundsCount > 0);

        int status = MA_PLAYING;
        bool done = false;
        if (all_reached)
        {
            status = MA_DONE;
            done = true;
            printf("team win\n");
        }
        else if (any_oob)
        {
            status = MA_DONE;
            done = true;
            printf("team lose\n");
        }

        // ===== 6) Package and return =====
        auto nextState = getGlobalState();
        auto doneTensor = torch::full({1, 1}, done ? 1.0 : 0.0, torch::kF64);
        return {nextState, rewardVector, doneTensor, status};
    }

    // Compute distance between a & b
    double distance(const Eigen::Vector2d &a, const Eigen::Vector2d &b)
    {
        return (a - b).norm();
    }

    void reset()
    {
        for (auto &p : positions_)
            {p.setZero();}
        std::fill(reached_.begin(), reached_.end(), 0);
    }
};
