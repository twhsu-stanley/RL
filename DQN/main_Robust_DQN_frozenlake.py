import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from utils_DQN import plot_frozenlake_dqn, plot_evaluation_return
from DQN_Agent import DQN_Agent

"""Robust DQN for frozenlake"""

if __name__ == "__main__":
    # Seed everything for reproducible results
    seed = 2000
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    # Create FrozenLake environment
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], 
                   max_episode_steps=100, map_name="4x4", is_slippery=is_slippery)
    gamma = 0.95

    n_episodes = 2000
    epsilon_init = 1.0
    learning_rate_init = 5e-4
    epsilon_lb = 0.01
    epsilon_decay_rate = 0.9995
    batch_size = 32
    replay_buffer_capacity = 4000
    Q_net_target_update_freq = 50
    R = 0.4

    n_trials = 1

    evaluation_return = []
    for i in range(n_trials):
        robust_dqn_agent = DQN_Agent(env,
                                     gamma,
                                     learning_rate_init,
                                     epsilon_init,
                                     epsilon_lb,
                                     epsilon_decay_rate,
                                     batch_size,
                                     replay_buffer_capacity,
                                     Q_net_target_update_freq,
                                     R)
        
        robust_dqn_agent.DQN_learning(n_episodes)

        evaluation_return.append(robust_dqn_agent.evaluation_return)

        # Plot value function and corresponding optimal policy
        if i == 0:
            plot_frozenlake_dqn(robust_dqn_agent, is_slippery=is_slippery, algorithm="DQN")

    plot_evaluation_return(evaluation_return)

    # Test the policies on uncertain transitions
    p = R
    G_robust = 0
    n_episodes_test = 100000
    for t in range(n_episodes_test):
        G_robust += robust_dqn_agent.DQN_sim_perturbed(p)
    G_robust = G_robust/n_episodes_test

    print(f"Robust DQN-agent MC return: G_robust = {G_robust}")

    #filename = f"DQN/Robust_DQN_frozenlake.pkl"
    #with open(filename, "wb") as f:
    #    pickle.dump(evaluation_return, f)