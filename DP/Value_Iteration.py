import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

def value_iteration(env, gamma):
    n_state = env.observation_space.n
    n_action = env.action_space.n

    epsilon = 1e-8
    convergence = False
    Q_0 = np.zeros((n_state, n_action)) # initial Q function
    V_hist = []
    while not convergence:
        Q_plus = np.zeros((n_state, n_action))
        for state in range(n_state):
            for action in range(n_action):
                for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                    Q_plus[state, action] += prob * (reward + gamma * np.max(Q_0[next_state,:])) # Bellman Optimality Operator
        V_hist.append(np.max(Q_plus, axis=1))
        convergence = True if np.max(np.abs(Q_plus - Q_0)) < epsilon else False
        Q_0 = Q_plus.copy()

    # Optimal policy and value function
    Q_star = Q_plus
    V_star = np.max(Q_star, axis=1)
    policy_star = np.argmax(Q_star, axis=1)
    
    return V_star, np.array(V_hist), policy_star

def plot_value_and_policy(env, V, policy, **kwargs):
    """
    Plot value function and corresponding optimal policy on the state space
    """

    grid_size = env.unwrapped.desc.shape
    desc = env.unwrapped.desc.astype(str)

    fig1, ax1 = plt.subplots(figsize=(5,5))
    fig2, ax2 = plt.subplots(figsize=(5,5))
    fig2, ax3 = plt.subplots(figsize=(5,5))
    
    # Grid the state space
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if desc[i,j] == 'S':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black', alpha=0.35)
            elif desc[i,j] == 'H':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black', alpha=0.35)
            elif desc[i,j] == 'G':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black', alpha=0.35)
            else:
                rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black')
            ax1.add_patch(rect)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if desc[i,j] == 'S':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black', alpha=0.35)
            elif desc[i,j] == 'H':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black', alpha=0.35)
            elif desc[i,j] == 'G':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black', alpha=0.35)
            else:
                rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black')
            ax2.add_patch(rect)
    
    # Grid the state space
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if desc[i,j] == 'S':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black', alpha=0.35)
                ax3.text(j + 0.5, i + 0.5, "Start", color='black', fontsize=10, ha='center', va='center')
            elif desc[i,j] == 'H':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black', alpha=0.35)
                ax3.text(j + 0.5, i + 0.5, "Hole", color='black', fontsize=10, ha='center', va='center')
            elif desc[i,j] == 'G':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black', alpha=0.35)
                ax3.text(j + 0.5, i + 0.5, "Goal", color='black', fontsize=10, ha='center', va='center')
            else:
                rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black')
                ax3.text(j + 0.5, i + 0.5, "Frozen", color='black', fontsize=10, ha='center', va='center')
            ax3.add_patch(rect)
    
    # action arrows: 0: Left, 1: Down, 2: Right, 3: Up
    arrow_dict = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)} # upside down y-axis
    
    # Write value function in the center of each box
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state = i * grid_size[1] + j
            # Value function numerical values
            value = round(V[state], 3)
            ax1.text(j + 0.5, i + 0.5, str(value), color='blue',
                    fontsize=12, ha='center', va='center')
            
            # Policy arrows
            action = policy[state]
            dx, dy = arrow_dict[action]
            if desc[i,j] in ['H','G']: # hole or goal
                continue
            ax2.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    # Set limits and labels
    ax1.set_xlim(0, grid_size[1])
    ax1.set_ylim(0, grid_size[0])
    ax1.set_aspect('equal')
    ax1.invert_yaxis()  # top-left is (0,0)
    ax1.set_xticks(np.arange(grid_size[1]+1))
    ax1.set_yticks(np.arange(grid_size[0]+1))
    ax1.grid(False)
    ax1.set_title("V(s): {}, is_slippery={}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False)))

    # Set limits and labels
    ax2.set_xlim(0, grid_size[1])
    ax2.set_ylim(0, grid_size[0])
    ax2.set_aspect('equal')
    ax2.invert_yaxis()  # top-left is (0,0)
    ax2.set_xticks(np.arange(grid_size[1]+1))
    ax2.set_yticks(np.arange(grid_size[0]+1))
    ax2.grid(False)
    ax2.set_title("Trained Policy: {}, is_slippery={}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False)))

    # Set limits and labels
    ax3.set_xlim(0, grid_size[1])
    ax3.set_ylim(0, grid_size[0])
    ax3.set_aspect('equal')
    ax3.invert_yaxis()  # top-left is (0,0)
    ax3.set_xticks(np.arange(grid_size[1]+1))
    ax3.set_yticks(np.arange(grid_size[0]+1))
    ax3.grid(False)
    ax3.set_title("Map")
    plt.show()

def simulate_policy(env, policy, **kwargs):
    env.reset()
    reach_goal = False
    while not reach_goal:
        # restart a new episode until the player reaches the goal
        state, info = env.reset() # starting state at 0
        state_hist = [state]
        action_hist = []
        while True:
            action = policy[state]
            state, reward, done, truncated, info = env.step(action)
            state_hist.append(state)
            action_hist.append(action)
            if state == 15: # 15 is the goal state
                reach_goal = True
                break
            if done or truncated:
                break

    print("State history ({}, is_slippery={}): {}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False), state_hist))
    print("Action history ({}, is_slippery={}): {}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False), action_hist))

    # Plot the trajectory on the map
    grid_size = env.unwrapped.desc.shape
    desc = env.unwrapped.desc.astype(str)

    fig, ax = plt.subplots(figsize=(5,5))
    
    # Grid the state space
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if desc[i,j] == 'S':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black', alpha=0.35)
            elif desc[i,j] == 'H':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black', alpha=0.35)
            elif desc[i,j] == 'G':
                rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black', alpha=0.35)
            else:
                rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black')
            ax.add_patch(rect)

    # Plot trajectory
    # action arrows: 0: Left, 1: Down, 2: Right, 3: Up
    #arrow_dict = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)} # upside down y-axis

    episode_length = len(state_hist)
    for k in range(episode_length-1):
        state = state_hist[k]
        state_next = state_hist[k+1]
        action = action_hist[k]
        assert action == policy[state]
        
        #dx, dy = arrow_dict[action]
        i_k, j_k = state // grid_size[1], state % grid_size[1]
        i_next, j_next = state_next // grid_size[1], state_next % grid_size[1]

        if i_k == i_next and j_k == j_next:
            continue # no movement
        ax.arrow(j_k + 0.5, i_k + 0.5, (j_next - j_k)*0.85, (i_next - i_k)*0.85, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax.arrow(j_k + 0.5, i_k + 0.5, (j_next - j_k)*0.85, (i_next - i_k)*0.85, head_width=0.1, head_length=0.1, fc='red', ec='red')
        #ax.arrow(j_k + 0.5, i_k + 0.5, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        plt.scatter(j_k + 0.5, i_k + 0.5, color='red', s=50)

    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_aspect('equal')
    ax.invert_yaxis()  # top-left is (0,0)
    ax.set_xticks(np.arange(grid_size[1]+1))
    ax.set_yticks(np.arange(grid_size[0]+1))
    ax.grid(False)
    ax.set_title("Sim Trajectory: {}, is_slippery={}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False)))
    plt.show()

    return state_hist, action_hist

if __name__ == "__main__":
    # Create FrozenLake environmentV
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False#True#
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)

    gamma = 0.95

    V_star, V_hist, policy_star = value_iteration(env, gamma)

    plt.figure()
    plt.plot(range(V_hist.shape[0]), np.mean(V_hist, axis=1))
    #plt.plot(range(V_hist.shape[0]), np.max(V_hist, axis=1))
    plt.xlabel('Value Iteration Steps')
    plt.ylabel('Average Value Function')
    plt.title('Value Iteration, is_slippery={}'.format(is_slippery))
    plt.grid(True)
    #plt.show()

    # Plot value function and optimal policy
    plot_value_and_policy(env, V_star, policy_star, algorithm="Value Iteration", is_slippery=is_slippery)

    # Simulate a trajectory using the optimal policy
    simulate_policy(env, policy_star, algorithm="Value Iteration", is_slippery=is_slippery)