# Robust RL Numerical Experiments

This repository contains code for the numerical experiments in the project "Uncertainty-Aware Robust Reinforcement Learning for Robotic Motion Planning under Model Uncertainty", focusing on robust reinforcement learning under transition-model uncertainty. 
Only the `Tabular/` and `DQN/` folders are needed to reproduce the results in the report.

## Code organization

### Robust Q-learning on FrozenLake

- Core implementation: `Tabular/Tabular_Agent.py`
  - The robust Q-learning algorithm is implemented in the `Tabular_Agent` class, especially the `Robust_Q_learning(...)` method.
- Main experiment script: `Tabular/main_Robust_Q_frozenlake.py`
  - Trains robust Q-learning on FrozenLake.
  - Evaluates the learned robust policy against regular Q-learning under perturbed transitions.
  - Saves learning-curve data as `Tabular/Robust_Q_frozenlake_R{R}_C{C}.pkl`.
- Plot script: `Tabular/plot_Robust_Q.py`
  - Loads saved `.pkl` files and generates the FrozenLake plots used in the report.

### Robust DQN on CartPole

- Core implementation: `DQN/DQN_Agent.py`
  - The robust DQN algorithm is implemented in the `DQN_Agent` class, especially the `DQN_learning(...)` method when using nonzero robustness parameters `R` and `C`.
- Main experiment script: `DQN/main_Robust_DQN_cartpole.py`
  - Trains robust DQN on CartPole.
  - Evaluates the learned robust policy against regular DQN under perturbed transitions.
  - Saves learning-curve data as `DQN/Robust_DQN_cartpole_R{R}_C{C}.pkl`.
- Plot script: `DQN/plot_Robust_DQN.py`
  - Loads saved `.pkl` files and generates the CartPole plots used in the report.

## Reproducing the results

Run all commands from the repository root.

### 1. Install dependencies

The experiments use Python with `numpy`, `matplotlib`, `gymnasium`, and `torch`.

```bash
pip install numpy matplotlib gymnasium torch
```

### 2. Reproduce FrozenLake robust Q-learning results

Edit the `R` and `C` variables in `Tabular/main_Robust_Q_frozenlake.py`, then run:

```bash
python Tabular/main_Robust_Q_frozenlake.py
```

To reproduce the FrozenLake plots, generate the saved `.pkl` files for the parameter settings used by `Tabular/plot_Robust_Q.py`, including:

- fixed `C = 1.0` with `R = 0.0, 0.1, 0.2, 0.4`
- fixed `R = 0.2` with `C = 0.0, 1.0, 1.5, 2.0`

Then run:

```bash
python Tabular/plot_Robust_Q.py
```

This generates the FrozenLake comparison plots in the `Tabular/` folder.

### 3. Reproduce CartPole robust DQN results

Edit the `R` and `C` variables in `DQN/main_Robust_DQN_cartpole.py`, then run:

```bash
python DQN/main_Robust_DQN_cartpole.py
```

To reproduce the CartPole plots, generate the saved `.pkl` files for the parameter settings used by `DQN/plot_Robust_DQN.py`, including:

- fixed `R = 0.1` with `C = 0, 0.05, 0.1, 0.2`
- fixed `C = 0.05` with the saved robust-DQN runs compared in the plot script

Then run:

```bash
python DQN/plot_Robust_DQN.py
```

This generates the CartPole learning-curve plots.

## Notes

- The user need to manually set `R` and `C` in the main scripts.
- Each main script saves the training results as a `.pkl` file. The plotting scripts assume those files already exist.
- Runtime can be significant because the experiments average over multiple trials and run many evaluation episodes.
