import numpy as np
import pickle
import gymnasium as gym
from agent import Agent

"""Off-policy TD(0) Q-learning control"""

if __name__ == "__main__":
    # Create FrozenLake environment
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False#True#
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)

    gamma = 0.95
    n_episodes = 80000
    epsilon = 1.0
    learning_rate = 0.1

    q_agent = Agent(env, gamma, learning_rate, epsilon)
    evaluation_return = q_agent.q_learning(n_episodes)
    q_agent.plot_value_and_policy(is_slippery=is_slippery, algorithm="Q-Learning")

    filename = f"Q_learning_slippery_{is_slippery}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(q_agent.evaluation_return, f)
