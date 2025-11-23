import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils_PG import plot_evaluation_return

# Compare the evaluation returns from different algorithms
with open("Policy_Gradient/ACTOR-CRITIC_frozenlake.pkl", "rb") as f:
    evaluation_return_fl_ac = pickle.load(f)

plot_evaluation_return(evaluation_return_fl_ac, algorithm = "ACTOR-CRITIC")

with open("Policy_Gradient/REINFORCE_frozenlake.pkl", "rb") as f:
    evaluation_return_fl_r = pickle.load(f)

plot_evaluation_return(evaluation_return_fl_r, algorithm = "REINFORCE")