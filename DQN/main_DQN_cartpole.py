import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from utils_DQN import plot_evaluation_return, plot_cartpole_angles
from DQN_Agent import DQN_Agent

"""DQN for cartpole"""

if __name__ == "__main__":
    # Seed everything for reproducible results
    seed = 2000
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")

    gamma = 0.9
    n_episodes = 1000
    epsilon_init = 0.999
    learning_rate_init = 1e-3
    epsilon_lb = 0.01
    epsilon_decay_rate = 0.999
    batch_size = 64
    replay_buffer_capacity = 4000
    Q_net_target_update_freq = 100

    n_trials = 30

    evaluation_return = []
    for i in range(n_trials):
        dqn_agent = DQN_Agent(env,
                            gamma,
                            learning_rate_init,
                            epsilon_init,
                            epsilon_lb,
                            epsilon_decay_rate,
                            batch_size,
                            replay_buffer_capacity,
                            Q_net_target_update_freq)
        
        dqn_agent.DQN_learning(n_episodes)

        evaluation_return.append(dqn_agent.evaluation_return)

        if i == 0:
            plot_evaluation_return(evaluation_return)

            # Simulation using the learned policy
            # NOTE: need to set epsilon to 0
            angle_hist = []
            for s in range(30):
                state_hist, action_hist, reward_hist = dqn_agent.DQN_sim()
                state_hist = np.array(state_hist)
                angle_hist.append(state_hist[:,2])
            plot_cartpole_angles(angle_hist)

    plot_evaluation_return(evaluation_return)

    cartpole = {"evaluation_return":evaluation_return, "angle_hist": angle_hist}
    filename = f"DQN/DQN_cartpole_1.pkl"
    with open(filename, "wb") as f:
        pickle.dump(cartpole, f)