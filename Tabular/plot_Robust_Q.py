import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils_tabular import calc_evaluation_return_mean_std
import gymnasium as gym

# Plot frozenlake
with open("Tabular/Robust_Q_frozenlake_R0.pkl", "rb") as f:
    evaluation_return_0 = pickle.load(f)
mean_0, std_0, _ = calc_evaluation_return_mean_std(evaluation_return_0)

with open("Tabular/Robust_Q_frozenlake_R0.1.pkl", "rb") as f:
    evaluation_return_01 = pickle.load(f)
mean_01, std_01, _ = calc_evaluation_return_mean_std(evaluation_return_01)

with open("Tabular/Robust_Q_frozenlake_R0.2.pkl", "rb") as f:
    evaluation_return_02 = pickle.load(f)
mean_02, std_02, _ = calc_evaluation_return_mean_std(evaluation_return_02)

with open("Tabular/Robust_Q_frozenlake_R0.4.pkl", "rb") as f:
    evaluation_return_04 = pickle.load(f)
mean_04, std_04, _ = calc_evaluation_return_mean_std(evaluation_return_04)

plt.figure()
plt.plot(mean_0, label = "R = 0")
plt.fill_between(range(len(mean_0)), mean_0 - std_0, mean_0 + std_0, alpha=0.2)
plt.plot(mean_01, label = "R = 0.1")
plt.fill_between(range(len(mean_01)), mean_01 - std_01, mean_01 + std_01, alpha=0.2)
plt.plot(mean_02, label = "R = 0.2")
plt.fill_between(range(len(mean_02)), mean_02 - std_02, mean_02 + std_02, alpha=0.2)
plt.plot(mean_04, label = "R = 0.4")
plt.fill_between(range(len(mean_04)), mean_04 - std_04, mean_04 + std_04, alpha=0.2)
plt.grid()
plt.xlim(0, 30000)
plt.ylim(0, 0.8)
plt.xlabel("Culmulative Time Steps")
plt.ylabel("Evaluation Return: V(initial state)")
plt.title("Learning Curve of Robust Q-Learning on FrozenLake")
plt.legend(loc="upper right")
plt.show()