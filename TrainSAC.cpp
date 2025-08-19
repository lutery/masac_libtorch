#include <fstream>
#include <Eigen/Core>
#include <torch/torch.h>
#include <random>
#include "SoftActorCritic.h"
#include "Models.h"
#include "TestEnvironment.h"
#include "ReplayBuffer.h"
#include "MultiAgentEnvironment.h"
#include "MultiSoftActorCritic.h"

int main()
{
    // Random number generators for goal placement
    std::random_device rd;
    std::mt19937 randomEngine(rd());
    std::uniform_int_distribution<> uniformInt(-5, 5);

    // ===== Environment =====
    const int numberOfAgents = 3;
    std::vector<Eigen::Vector2d> initialGoals(numberOfAgents);
    for (int i = 0; i < numberOfAgents; ++i)
    {
        initialGoals[i](0) = static_cast<double>(uniformInt(randomEngine));
        initialGoals[i](1) = static_cast<double>(uniformInt(randomEngine));
    }
    MultiAgentEnvironment env(numberOfAgents, initialGoals);

    // Multiple SAC Models.
    const int local_obs_dim = 4; // [pos_x, pos_y, goal_x, goal_y]
    const int action_dim = 2;    // action = (ax, ay) in [-1, 1] via tanh in Actor
    const double alpha = 0.1;
    const double gamma = 0.99;
    const double tau = 0.005;
    const double lr = 1e-3;
    const double alpha_lr = 1e-3;

    MultiAgentSAC masac(numberOfAgents, local_obs_dim, action_dim, alpha, alpha_lr, gamma, tau, lr);

    // Replay buffer
    ReplayBuffer buffer(100000);

    // Training loop.
    const uint n_iter = 2500;
    const uint n_epochs = 100;
    const uint batch_size = 512;
    const uint mini_batch_size = 128;

    // ===== Logging =====
    std::ofstream csv("../data/ma_independent_q.csv");
    csv << "epoch";
    for (int i = 0; i < numberOfAgents; ++i)
        csv << ",pos_x" << i << ",pos_y" << i << ",goal_x" << i << ",goal_y" << i;
    for (int i = 0; i < numberOfAgents; ++i)
        csv << ",reward" << i;
    csv << ",done,status\n";

    // reward per epoch
    std::ofstream epoch_txt("../data/ma_epoch_rewards.txt",std::ios::out | std::ios::trunc);
    epoch_txt << "epoch";
    for (int i = 0; i < numberOfAgents; ++i)
        epoch_txt << " reward_sum_" << i;
    epoch_txt << " reward_sum_all\n";
    epoch_txt.close(); 

    double bestAverageReward = -1e18;

    // ===== Training Loop =====
    for (auto epoch = 0; epoch < n_epochs; ++epoch)
    {
        double reward_sum_all = 0.0;
        std::vector<double> reward_sum_per_agent(numberOfAgents, 0.0);

        printf("Epoch %u/%u\n", epoch + 1, n_epochs);
        double averageRewardEpoch = 0.0;

        for (uint64_t t = 0; t < n_iter; ++t)
        {
            //- 1) Prepare globalstate and per-agent local observations
            auto globalSate = env.getGlobalState();

            std::vector<torch::Tensor> localObservations(numberOfAgents);
            for (int i = 0; i < numberOfAgents; i++)
            {
                localObservations[i] = env.getLocalObservation(i);
            }

            //- 2) Sample joint action from current policy (stochastic)
            auto jointAction = masac.act(localObservations); // [1, 2*N]

            // 3) Interact with the environment
            auto [nextGlobalState, rewardVector, doneFlag, status] = env.Step(jointAction);

            // 4) Push transition into replay buffer note: rewardVector has shape [1, N]
            buffer.add(globalSate, jointAction, rewardVector, nextGlobalState, doneFlag);
            // Accumulate epoch-average reward (mean of per-agent rewards here)
            averageRewardEpoch += rewardVector.mean().item<double>() / static_cast<double>(n_epochs);
            // save reward to csv file
            for (int i = 0; i < numberOfAgents; ++i)
            {
                double ri = rewardVector[0][i].item<double>();
                reward_sum_per_agent[i] += ri;
                reward_sum_all += ri;
            }

            // 5) Learn
            if (buffer.size() > batch_size)
            {
                masac.trainStep(buffer, mini_batch_size);
            }
            // 6) Episode termination handling
            if (doneFlag.item<double>() == 1.0)
            {
                // Re-initialize new goals and reset positions
                for (int i = 0; i < numberOfAgents; ++i)
                {
                    env.goals_[i](0) = static_cast<double>(uniformInt(randomEngine));
                    env.goals_[i](1) = static_cast<double>(uniformInt(randomEngine));
                }
                env.reset();
            }

            // 7) CSV logging for analysis
            csv << epoch;
            for (int i = 0; i < numberOfAgents; ++i)
                csv << "," << env.positions_[i](0)
                    << "," << env.positions_[i](1)
                    << "," << env.goals_[i](0)
                    << "," << env.goals_[i](1);
            for (int i = 0; i < numberOfAgents; ++i)
                csv << "," << rewardVector[0][i].item<double>();
            csv << "," << doneFlag.item<double>() << "," << status << "\n";
        }

        if (averageRewardEpoch > bestAverageReward)
        {
            bestAverageReward = averageRewardEpoch;
            std::cout << "[Epoch " << epoch << "] New best average reward = " << bestAverageReward << "\n";
        }
        // Optional: randomize for the next epoch as well
        for (int i = 0; i < numberOfAgents; ++i)
        {
            env.goals_[i](0) = static_cast<double>(uniformInt(randomEngine));
            env.goals_[i](1) = static_cast<double>(uniformInt(randomEngine));
        }
        env.reset();

        // reward logging
        epoch_txt.open("../data/ma_epoch_rewards.txt",std::ios::app);
        epoch_txt << epoch;
        for (int i = 0; i < numberOfAgents; ++i)
            epoch_txt << " " << reward_sum_per_agent[i];
        epoch_txt << " " << averageRewardEpoch << "\n";
        epoch_txt.close();
    }

    csv.close();
    return 0;
}
