import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from hw1_value_iteration import plot_value_and_policy, simulate_policy

def policy_evaluation(env, policy, gamma):
    n_state = env.observation_space.n
    V = np.zeros(n_state)
    epsilon = 1e-8
    convergence = False
    while not convergence:
        V_old = V.copy()
        for state in range(n_state):
            action = policy[state]
            V[state] = 0
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                V[state] += prob * (reward + gamma * V_old[next_state])
        convergence = True if np.max(np.abs(V - V_old)) < epsilon else False
    return V

def policy_improvement(env, V):
    n_state = env.observation_space.n
    n_action = env.action_space.n

    policy_greedy = np.zeros(n_state)
    Q = np.zeros((n_state, n_action))
    for state in range(n_state):
        for action in range(n_action):
            Q[state, action] = 0
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                Q[state, action] += prob * (reward + gamma * V[next_state])
        policy_greedy[state] = np.argmax(Q[state, :])
    return policy_greedy

def policy_iteration(env, policy_init, gamma):
    policy = policy_init
    termination = False
    V_hist = []
    while not termination:
        V = policy_evaluation(env, policy, gamma)
        V_hist.append(V)
        policy_greedy = policy_improvement(env, V)
        if np.array_equal(policy, policy_greedy):
            termination = True
        else:
            policy = policy_greedy.copy()
    V_hist = np.array(V_hist)
    policy_star = policy.copy()
    return policy_star, V_hist

if __name__ == "__main__":
    # Create FrozenLake environment
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)

    gamma = 0.95

    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy_init = np.zeros(n_state) 
    policy_star, V_hist = policy_iteration(env, policy_init, gamma)

    plt.figure()
    plt.plot(range(V_hist.shape[0]), np.mean(V_hist, axis=1))
    #plt.plot(range(V_hist.shape[0]), np.max(V_hist, axis=1))
    plt.xlabel('Policy Iteration Steps')
    plt.ylabel('Average Value Function')
    plt.grid(True)
    plt.title('Policy Iteration, is_slippery={}'.format(is_slippery))
    #plt.show()

    # Plot value function and optimal policy
    plot_value_and_policy(env, V_hist[-1,:], policy_star, algorithm="Policy Iteration", is_slippery=is_slippery)

    # Simulate a trajectory using the optimal policy
    simulate_policy(env, policy_star, algorithm="Policy Iteration", is_slippery=is_slippery)