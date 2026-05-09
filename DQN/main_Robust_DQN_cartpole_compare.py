import os
import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from DQN_Agent import DQN_Agent
from utils_DQN import calc_evaluation_return_mean_std

"""Compare Robust DQN learning curves on CartPole for selected R-C settings."""


def format_float(value):
    return f"{value:g}"


def seed_everything(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


def train_robust_dqn_trials(config, hyperparams, n_trials, base_seed):
    R, C = config
    evaluation_return = []

    for trial in range(n_trials):
        trial_seed = base_seed + 1000 * trial + int(10000 * R) + int(100000 * C)
        seed_everything(trial_seed)

        env = gym.make("CartPole-v1", max_episode_steps=100)
        env.reset(seed=trial_seed)
        env.action_space.seed(trial_seed)
        env.observation_space.seed(trial_seed)

        print(
            f"Training Robust DQN: R={format_float(R)}, C={format_float(C)}, "
            f"trial {trial + 1}/{n_trials}"
        )

        robust_dqn_agent = DQN_Agent(
            env,
            hyperparams["gamma"],
            hyperparams["learning_rate_init"],
            hyperparams["epsilon_init"],
            hyperparams["epsilon_lb"],
            hyperparams["epsilon_decay_rate"],
            hyperparams["batch_size"],
            hyperparams["replay_buffer_capacity"],
            hyperparams["Q_net_target_update_freq"],
            R,
            C,
            hyperparams["n_uncertainty_samples"],
        )

        robust_dqn_agent.DQN_learning(hyperparams["n_episodes"])
        evaluation_return.append(robust_dqn_agent.evaluation_return)
        env.close()

    return evaluation_return


def plot_learning_curves(results, configs, title, output_path):
    plt.figure(figsize=(8, 5))

    for R, C in configs:
        evaluation_return = results[(R, C)]
        mean, std, _ = calc_evaluation_return_mean_std(evaluation_return)
        x = np.arange(len(mean))
        label = f"R={format_float(R)}, C={format_float(C)}"

        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.grid()
    plt.xlabel("Cumulative Time Steps")
    plt.ylabel("Evaluation Return V(initial state)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    base_seed = 2000
    seed_everything(base_seed)

    hyperparams = {
        "gamma": 0.90,
        "n_episodes": 900,
        "epsilon_init": 0.999,
        "learning_rate_init": 5e-4,
        "epsilon_lb": 0.01,
        "epsilon_decay_rate": 0.999,
        "batch_size": 64,
        "replay_buffer_capacity": 4000,
        "Q_net_target_update_freq": 100,
        "n_uncertainty_samples": 16,
    }

    n_trials = 10

    fixed_R_configs = [(0.1, 0.0), (0.1, 0.05), (0.1, 0.1)]
    fixed_C_configs = [(0.1, 0.05), (0.2, 0.05)]
    configs = list(dict.fromkeys(fixed_R_configs + fixed_C_configs))

    results = {}
    for config in configs:
        R, C = config
        evaluation_return = train_robust_dqn_trials(
            config,
            hyperparams,
            n_trials,
            base_seed,
        )
        results[config] = evaluation_return

        filename = (
            "DQN/"
            f"Robust_DQN_cartpole_R{format_float(R)}_C{format_float(C)}_"
            f"ntrials{n_trials}.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(evaluation_return, f)

    comparison = {
        "hyperparams": hyperparams,
        "n_trials": n_trials,
        "fixed_R_configs": fixed_R_configs,
        "fixed_C_configs": fixed_C_configs,
        "results": results,
    }
    with open("DQN/Robust_DQN_cartpole_learning_curve_comparison.pkl", "wb") as f:
        pickle.dump(comparison, f)

    plot_learning_curves(
        results,
        fixed_R_configs,
        "Robust DQN CartPole Learning Curves: R=0.1",
        "DQN/Robust_DQN_cartpole_fixed_R0.1_compare_C.png",
    )
    plot_learning_curves(
        results,
        fixed_C_configs,
        "Robust DQN CartPole Learning Curves: C=0.05",
        "DQN/Robust_DQN_cartpole_fixed_C0.05_compare_R.png",
    )

    plt.show()
