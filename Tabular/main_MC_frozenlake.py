import numpy as np
import pickle
import gymnasium as gym
from Tabular_Agent import Tabular_Agent
from utils_tabular import plot_frozenlake_tabular

"""On-policy first-visit MC control"""

if __name__ == "__main__":
    # Create FrozenLake environment
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False#True#
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)

    gamma = 0.95
    if is_slippery:
        n_episodes = 250000
    else:
        n_episodes = 100000
    learning_rate_init = 0.1

    mc_agent = Tabular_Agent(env, gamma, learning_rate_init)
    mc_agent.MC_Control(n_episodes)
    plot_frozenlake_tabular(mc_agent, is_slippery=is_slippery, algorithm="MC Control")

    filename = f"MC_slippery_{is_slippery}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(mc_agent.evaluation_return, f)