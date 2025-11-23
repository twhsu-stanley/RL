import numpy as np
import gymnasium as gym

def is_first_visit(state_hist, action_hist, t):
    for i in range(t):
        if state_hist[i] == state_hist[t] and action_hist[i] == action_hist[t]:
            return False
    return True

class Tabular_Agent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float,
        learning_rate_init: float,
        lr_decay_rate: float = 0.999,
        episode_start_decay_lr: int = 5000,
        epsilon_init: float = 1.0,
        epsilon_lb: float = 0.01,
        epsilon_decay_rate: float = 0.999,
    ):
        self.env = env
        self.gamma = gamma
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

        self.Q = np.zeros((self.n_state, self.n_action))

        self.learning_rate = learning_rate_init
        self.lr_decay_rate = lr_decay_rate
        self.episode_start_decay_lr = episode_start_decay_lr

        self.epsilon_init = epsilon_init
        self.epsilon = epsilon_init
        self.epsilon_lb = epsilon_lb
        self.epsilon_decay_rate = epsilon_decay_rate

        self.evaluation_return = []
        self.evaluation_return.append(np.max(self.Q[0, :]))

    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_action)
        else:
            return np.argmax(self.Q[state, :])

    def MC_Control(self, n_episodes):
        """On-policy first-visit MC control algorithm"""

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")

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
                
                if done or truncated:
                    break

            G = 0
            for t in reversed(range(len(action_hist))):
                G = self.gamma * G + reward_hist[t]

                # check first visit
                if is_first_visit(state_hist, action_hist, t):
                    self.Q[state_hist[t], action_hist[t]] += self.learning_rate * (G - self.Q[state_hist[t], action_hist[t]])
                    self.evaluation_return.append(np.max(self.Q[0, :]))

            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

            # Schedule learning rate decay
            if episode >= self.episode_start_decay_lr:
                self.learning_rate = self.learning_rate * self.lr_decay_rate

        Q_star = self.Q.copy()
        policy_star = np.argmax(Q_star, axis=1)
        
        return Q_star, policy_star
    
    def SARSA(self, n_episodes):
        """On-policy TD(0) Sarsa control"""

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")
            
            state, info = self.env.reset() # starting state at 0

            action = self.epsilon_greedy_policy(state, self.epsilon)
            
            while True:
                state_plus, reward, done, truncated, info = self.env.step(action)

                action_plus = self.epsilon_greedy_policy(state_plus, self.epsilon)

                self.Q[state, action] += self.learning_rate * (reward + self.gamma * self.Q[state_plus, action_plus] - self.Q[state, action])

                state = state_plus
                action = action_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                if done or truncated:
                    break
            
            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

            # Schedule learning rate decay
            if episode >= self.episode_start_decay_lr:
                self.learning_rate = self.learning_rate * self.lr_decay_rate

    def Q_learning(self, n_episodes):
        """Off-policy TD(0) Q-learning control"""

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")
            
            state, info = self.env.reset() # starting state at 0

            while True:
                action = self.epsilon_greedy_policy(state, self.epsilon)

                state_plus, reward, done, truncated, info = self.env.step(action)

                self.Q[state, action] += self.learning_rate * (reward + self.gamma * np.max(self.Q[state_plus,:]) - self.Q[state, action]) 
                
                state = state_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                if done or truncated:
                    break
            
            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

            # Schedule learning rate decay
            if episode >= self.episode_start_decay_lr:
                self.learning_rate = self.learning_rate * self.lr_decay_rate

        