import matplotlib.pyplot as plt
import pickle
from utils_DQN import calc_evaluation_return_mean_std


with open("DQN/Robust_DQN_cartpole_R0.1_C0.pkl", "rb") as f:
    evaluation_return_R01_C0 = pickle.load(f)
mean_R01_C0, std_R01_C0, _ = calc_evaluation_return_mean_std(evaluation_return_R01_C0)

with open("DQN/Robust_DQN_cartpole_R0.1_C0.05.pkl", "rb") as f:
    evaluation_return_R01_C005 = pickle.load(f)
mean_R01_C005, std_R01_C005, _ = calc_evaluation_return_mean_std(evaluation_return_R01_C005)

with open("DQN/Robust_DQN_cartpole_R0.1_C0.1.pkl", "rb") as f:
    evaluation_return_R01_C01 = pickle.load(f)
mean_R01_C01, std_R01_C01, _ = calc_evaluation_return_mean_std(evaluation_return_R01_C01)

with open("DQN/Robust_DQN_cartpole_R0.1_C0.2.pkl", "rb") as f:
    evaluation_return_R01_C02 = pickle.load(f)
mean_R01_C02, std_R01_C02, _ = calc_evaluation_return_mean_std(evaluation_return_R01_C02)

with open("DQN/Robust_DQN_cartpole_R0.2_C0.05.pkl", "rb") as f:
    evaluation_return_R02_C005 = pickle.load(f)
evaluation_return_R02_C005.pop(7)
mean_R02_C005, std_R02_C005, _ = calc_evaluation_return_mean_std(evaluation_return_R02_C005)

with open("DQN/Robust_DQN_cartpole_R0.4_C0.05.pkl", "rb") as f:
    evaluation_return_R04_C005 = pickle.load(f)
mean_R04_C005, std_R04_C005, _ = calc_evaluation_return_mean_std(evaluation_return_R04_C005)

plt.figure()
plt.plot(mean_R01_C0, label = "R = 0.1, C = 0")
plt.fill_between(range(len(mean_R01_C0)), mean_R01_C0 - std_R01_C0, mean_R01_C0 + std_R01_C0, alpha=0.2)
#plt.plot(mean_R01_C005, label = "R = 0.1, C = 0.05")
#plt.fill_between(range(len(mean_R01_C005)), mean_R01_C005 - std_R01_C005, mean_R01_C005 + std_R01_C005, alpha=0.2)
plt.plot(mean_R01_C01, label = "R = 0.1, C = 0.1")
plt.fill_between(range(len(mean_R01_C01)), mean_R01_C01 - std_R01_C01, mean_R01_C01 + std_R01_C01, alpha=0.2)
plt.plot(mean_R01_C02, label = "R = 0.1, C = 0.2")
plt.fill_between(range(len(mean_R01_C02)), mean_R01_C02 - std_R01_C02, mean_R01_C02 + std_R01_C02, alpha=0.2)
plt.grid()
plt.xlim(0, 35000)
plt.xlabel("Cumulative Time Steps")
plt.ylabel("Evaluation Return:  V(initial state)")
plt.title("Learning Curve of Robust DQN on Cartpole: R = 0.1")
plt.legend(loc="lower right")

plt.figure()
plt.plot(mean_R01_C0, label = "R = 0, C = 0.05")
plt.fill_between(range(len(mean_R01_C0)), mean_R01_C0 - std_R01_C0, mean_R01_C0 + std_R01_C0, alpha=0.2)
plt.plot(mean_R01_C005, label = "R = 0.1, C = 0.05")
plt.fill_between(range(len(mean_R01_C005)), mean_R01_C005 - std_R01_C005, mean_R01_C005 + std_R01_C005, alpha=0.2)
#plt.plot(mean_R02_C005, label = "R = 0.2, C = 0.05")
#plt.fill_between(range(len(mean_R02_C005)), mean_R02_C005 - std_R02_C005, mean_R02_C005 + std_R02_C005, alpha=0.2)
plt.plot(mean_R04_C005, label = "R = 0.4, C = 0.05")
plt.fill_between(range(len(mean_R04_C005)), mean_R04_C005 - std_R04_C005, mean_R04_C005 + std_R04_C005, alpha=0.2)
plt.grid()
plt.xlim(0, 35000)
plt.xlabel("Cumulative Time Steps")
plt.ylabel("Evaluation Return:  V(initial state)")
plt.title("Learning Curve of Robust DQN on Cartpole: C = 0.05")
plt.legend(loc="lower right")
plt.show()
