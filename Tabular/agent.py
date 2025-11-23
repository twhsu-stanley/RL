import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

def is_first_visit(state_hist, action_hist, t):
    for i in range(t):
        if state_hist[i] == state_hist[t] and action_hist[i] == action_hist[t]:
            return False
    return True

class Agent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float,
        learning_rate: float,
        epsilon_init: float,
    ):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_init = epsilon_init
        self.epsilon = epsilon_init

        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

        self.Q = np.zeros((self.n_state, self.n_action))

        self.evaluation_return = []
        self.evaluation_return.append(np.max(self.Q[0, :]))

    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_action)
        else:
            return np.argmax(self.Q[state, :])

    def simulate_episode_greedy(self):
        #reach_goal = False
        #while not reach_goal:
        state, info = self.env.reset() # starting state at 0
        state_hist = [state]
        action_hist = []
        reward_hist = []
        while True:
            # Select action using epsilon-greedy policy given current Q
            action = self.epsilon_greedy_policy(state, self.epsilon)
            action_hist.append(action)

            state, reward, done, truncated, info = self.env.step(action)
            state_hist.append(state)
            reward_hist.append(reward)
                
            if state == 15: # 15 is the goal state
                #reach_goal = True
                break
            if done or truncated:
                break
        return state_hist, action_hist, reward_hist

    def plot_value_and_policy(self, **kwargs):
        """
        Plot value function and corresponding optimal policy on the state space
        """

        grid_size = self.env.unwrapped.desc.shape
        desc = self.env.unwrapped.desc.astype(str)

        fig1, ax1 = plt.subplots(figsize=(5,5))
        fig2, ax2 = plt.subplots(figsize=(5,5))
        
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
        
        # action arrows: 0: Left, 1: Down, 2: Right, 3: Up
        arrow_dict = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)} # upside down y-axis
        
        # Write value function in the center of each box
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = i * grid_size[1] + j
                # Value function numerical values
                value = round(np.max(self.Q[state,:]), 2) # Bellman optimality of Q
                ax1.text(j + 0.5, i + 0.5, str(value), color='blue',
                        fontsize=12, ha='center', va='center')
                
                # Policy arrows
                action = np.argmax(self.Q[state,:]) # greedy policy w.r.t. Q
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

        plt.show()

    def mc_control(self, n_episodes):
        """On-policy first-visit MC control algorithm"""

        for episode in range(n_episodes):
            # Schedule epsilon decay to impose GLIE
            if episode >= 1:
                self.epsilon = max(self.epsilon_init * 0.9999 ** (episode-1), 0)
            else:
                self.epsilon = self.epsilon_init

            # Schedule learning rate decay to enhance stability
            if episode >= 5000:
                learning_rate = max(self.learning_rate * 0.9999 ** (episode-5000), 0.0001)
            else:
                learning_rate = self.learning_rate

            state_hist, action_hist, reward_hist = self.simulate_episode_greedy()

            G = 0
            for t in reversed(range(len(action_hist))):
                G = self.gamma * G + reward_hist[t]

                # check first visit
                if is_first_visit(state_hist, action_hist, t):
                    self.Q[state_hist[t], action_hist[t]] += learning_rate * (G - self.Q[state_hist[t], action_hist[t]])
                    self.evaluation_return.append(np.max(self.Q[0, :]))
        
        Q_star = self.Q.copy()
        policy_star = np.argmax(Q_star, axis=1) # TODO: check this
        
        return Q_star, policy_star
    
    def sarsa(self, n_episodes):
        """On-policy TD(0) Sarsa control"""

        for episode in range(n_episodes):
            # Schedule epsilon decay to impose GLIE
            if episode >= 1:
                self.epsilon = max(self.epsilon_init * 0.9999 ** (episode-1), 0)
            else:
                self.epsilon = self.epsilon_init

            # Schedule learning rate decay to enhance stability
            if episode >= 5000:
                learning_rate = max(self.learning_rate * 0.9999 ** (episode-5000), 0.001)
            else:
                learning_rate = self.learning_rate
            
            state, info = self.env.reset() # starting state at 0

            action = self.epsilon_greedy_policy(state, self.epsilon)
            
            while True:
                state_plus, reward, done, truncated, info = self.env.step(action)

                action_plus = self.epsilon_greedy_policy(state_plus, self.epsilon)

                self.Q[state, action] += learning_rate * (reward + self.gamma * self.Q[state_plus, action_plus] - self.Q[state, action])

                state = state_plus
                action = action_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                if state == 15 or done or truncated:
                    break

    def q_learning(self, n_episodes):
        """Off-policy TD(0) Q-learning control"""

        for episode in range(n_episodes):
            # Schedule epsilon decay to impose GLIE
            if episode >= 1:
                self.epsilon = max(self.epsilon_init * 0.9999 ** (episode-1), 0)
            else:
                self.epsilon = self.epsilon_init

            # Schedule learning rate decay to enhance stability
            if episode >= 5000:
                learning_rate = max(self.learning_rate * 0.9999 ** (episode-5000), 0.001)
            else:
                learning_rate = self.learning_rate
            
            state, info = self.env.reset() # starting state at 0

            while True:
                action = self.epsilon_greedy_policy(state, self.epsilon)

                state_plus, reward, done, truncated, info = self.env.step(action)

                self.Q[state, action] += learning_rate * (reward + self.gamma * np.max(self.Q[state_plus,:]) - self.Q[state, action]) 
                
                state = state_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                if state == 15 or done or truncated:
                    break