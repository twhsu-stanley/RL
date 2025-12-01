import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from utils_DQN import plot_evaluation_return, plot_cartpole_angles
from DQN_Agent import DQN_Agent

"""Robust DQN for cartpole"""

if __name__ == "__main__":
    # Seed everything for reproducible results
    seed = 2000
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    gamma = 0.90
    n_episodes = 900
    epsilon_init = 0.999
    learning_rate_init = 5e-4 # 1e-3
    epsilon_lb = 0.01
    epsilon_decay_rate = 0.999
    batch_size = 64
    replay_buffer_capacity = 4000
    Q_net_target_update_freq = 100
    R = 0.4

    n_trials = 30

    evaluation_return = []
    for i in range(n_trials):
        env = gym.make("CartPole-v1", max_episode_steps = 100)

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

        if i == 0:
            plot_evaluation_return(evaluation_return)

            # Simulation using the learned policy
            # NOTE: need to set epsilon to 0
            angle_hist = []
            for s in range(30):
                state_hist, action_hist, reward_hist = robust_dqn_agent.DQN_sim()
                state_hist = np.array(state_hist)
                angle_hist.append(state_hist[:,2])
            plot_cartpole_angles(angle_hist)

    #plot_evaluation_return(evaluation_return)

    #filename = f"DQN/Robust_DQN_cartpole_R{R}.pkl"
    #with open(filename, "wb") as f:
    #    pickle.dump(evaluation_return, f)

    ###################################################
    # Regular DQN for comparison
    env = gym.make("CartPole-v1", max_episode_steps = 100)

    dqn_agent = DQN_Agent(env,
                          gamma,
                          learning_rate_init,
                          epsilon_init,
                          epsilon_lb,
                          epsilon_decay_rate,
                          batch_size,
                          replay_buffer_capacity,
                          Q_net_target_update_freq,
                          R=0.0)
    
    dqn_agent.DQN_learning(n_episodes)
    
    ###################################################
    # Test the policies on uncertain transitions
    p = R
    G_robust = 0
    G = 0
    angle_hist_robust = []
    angle_hist = []
    n_test = 10000
    for t in range(n_test):
        print(f"Episode {t+1}/{n_test}")
        G_robust_episode, state_hist_robust = robust_dqn_agent.DQN_sim_perturbed(p)
        angle_hist_robust.append(np.array(state_hist_robust)[:,2])
        G_robust += G_robust_episode
        
        G_episode, state_hist = dqn_agent.DQN_sim_perturbed(p)
        angle_hist.append(np.array(state_hist)[:,2])
        G += G_episode

    G_robust = G_robust/n_test
    G = G/n_test

    print(f"Robust DQN-agent MC return: G_robust = {G_robust}")
    print(f"DQN-agent MC return: G = {G}")
    print(f"G_robust - G = {G_robust - G}")

    plot_cartpole_angles(angle_hist[:30], algorithm = "DQN")
    plot_cartpole_angles(angle_hist_robust[:30], algorithm = "Robust DQN")

    plt.show()