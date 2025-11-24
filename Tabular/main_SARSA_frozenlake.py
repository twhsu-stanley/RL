import numpy as np
import pickle
import gymnasium as gym
from Tabular_Agent import Tabular_Agent
from utils_tabular import plot_frozenlake_tabular

"""On-policy TD(0) Sarsa control"""

if __name__ == "__main__":
    # Create FrozenLake environment
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False#True#
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)

    gamma = 0.95
    n_episodes = 80000
    learning_rate_init = 0.1

    sarsa_agent = Tabular_Agent(env, gamma, learning_rate_init)
    sarsa_agent.SARSA(n_episodes)
    plot_frozenlake_tabular(sarsa_agent, is_slippery=is_slippery, algorithm="SARSA")

    filename = f"SARSA_slippery_{is_slippery}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(sarsa_agent.evaluation_return, f)