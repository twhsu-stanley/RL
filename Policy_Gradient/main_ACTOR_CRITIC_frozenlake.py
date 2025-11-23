import numpy as np
import pickle
import matplotlib.pyplot as plt
import gymnasium as gym
from PG_Agent import PG_Agent
from utils_PG import plot_frozenlake_REINFORCE, plot_frozenlake_ACTOR_CRITIC, plot_evaluation_return

"""ACTOR-CRITIC for frozenlake"""

if __name__ == "__main__":
    # Create FrozenLake environment
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False

    env = gym.make('FrozenLake-v1', desc = ["SFFF", "FHFH", "FFFH", "HFFG"], map_name = "4x4",
                   max_episode_steps = 100, is_slippery = is_slippery)

    gamma = 0.95
    n_episodes = 2500
    lr_policy = 0.1
    lr_value = 7e-4
    n_trials = 30

    evaluation_return = []
    for i in range(n_trials):
        pg_agent = PG_Agent(env, gamma, learning_rate_policy=lr_policy, learning_rate_value=lr_value)
    
        pg_agent.ACTOR_CRITIC_learning(n_episodes)
        
        evaluation_return.append(pg_agent.evaluation_return)

        if i == 0:
            plot_frozenlake_ACTOR_CRITIC(pg_agent, is_slippery=is_slippery, algorithm="ACTOR-CRITIC")

    plot_evaluation_return(evaluation_return)

    filename = f"Policy_Gradient/ACTOR-CRITIC_frozenlake.pkl"
    with open(filename, "wb") as f:
        pickle.dump(evaluation_return, f)