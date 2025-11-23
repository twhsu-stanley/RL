import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils_DQN import plot_evaluation_return, plot_cartpole_angles
from DQN_Agent import DQN_Agent
import gymnasium as gym

# Plot frozenlake DQN
with open("DQN/DQN_frozenlake.pkl", "rb") as f:
    evaluation_return_fl_dqn = pickle.load(f)
plot_evaluation_return(evaluation_return_fl_dqn)

# Plot cartpole DQN
with open("DQN/DQN_cartpole_1.pkl", "rb") as f:
    cartpole = pickle.load(f)
plot_evaluation_return(cartpole["evaluation_return"])
plot_cartpole_angles(cartpole["angle_hist"])