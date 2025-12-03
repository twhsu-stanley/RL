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
        lr_init: float,
        step_start_decay_lr: int = 10000,
        epsilon_init: float = 1.0,
        epsilon_lb: float = 0.01,
        epsilon_decay_rate: float = 0.999,
        R: float = 0.0
    ):
        self.env = env
        self.gamma = gamma
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

        self.Q = np.zeros((self.n_state, self.n_action))

        # Learning rate parameters
        self.lr_init = lr_init
        self.lr = lr_init
        self.step_start_decay_lr = step_start_decay_lr

        # Epsilon-greedy policy parameters
        self.epsilon_init = epsilon_init
        self.epsilon = epsilon_init
        self.epsilon_lb = epsilon_lb
        self.epsilon_decay_rate = epsilon_decay_rate

        self.evaluation_return = []
        self.evaluation_return.append(np.max(self.Q[0, :]))

        ####################################################
        # Below are parameters for robust tabular RL
        self.R = R # for the R-contamination uncertainty set

    def epsilon_greedy_policy(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_action)
        else:
            return np.argmax(self.Q[state, :])

    def MC_Control(self, n_episodes):
        """On-policy first-visit MC control algorithm"""
        
        cum_step = 0

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")

            state, info = self.env.reset() # starting state at 0
            state_hist = [state]
            action_hist = []
            reward_hist = []
            while True:
                # Select action using epsilon-greedy policy given current Q
                action = self.epsilon_greedy_policy(state)
                action_hist.append(action)

                state, reward, done, truncated, info = self.env.step(action)
                state_hist.append(state)
                reward_hist.append(reward)
                
                if done or truncated:
                    break

            G = 0
            for t in reversed(range(len(action_hist))):
                cum_step += 1

                # Schedule learning rate decay
                if cum_step >= self.step_start_decay_lr:
                    self.lr = self.lr_init / (cum_step - self.step_start_decay_lr + 1)

                G = self.gamma * G + reward_hist[t]

                # check first visit
                if is_first_visit(state_hist, action_hist, t):
                    self.Q[state_hist[t], action_hist[t]] += self.lr * (G - self.Q[state_hist[t], action_hist[t]])
                    self.evaluation_return.append(np.max(self.Q[0, :]))

            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

        Q_star = self.Q.copy()
        policy_star = np.argmax(Q_star, axis=1)
        
        return Q_star, policy_star
    
    def SARSA(self, n_episodes):
        """On-policy TD(0) Sarsa control"""

        cum_step = 0

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")
            
            state, info = self.env.reset() # starting state at 0

            action = self.epsilon_greedy_policy(state)
            
            while True:
                state_plus, reward, done, truncated, info = self.env.step(action)

                action_plus = self.epsilon_greedy_policy(state_plus)

                self.Q[state, action] += self.lr * (reward + self.gamma * self.Q[state_plus, action_plus] - self.Q[state, action])

                state = state_plus
                action = action_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                if done or truncated:
                    break

                cum_step += 1

                # Schedule learning rate decay
                if cum_step >= self.step_start_decay_lr:
                    self.lr = self.lr_init / (cum_step - self.step_start_decay_lr + 1)
            
            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

    def Q_learning(self, n_episodes):
        """Off-policy TD(0) Q-learning control"""

        cum_step = 0

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")
            
            state, info = self.env.reset() # starting state at 0

            while True:
                action = self.epsilon_greedy_policy(state)

                state_plus, reward, done, truncated, info = self.env.step(action)

                self.Q[state, action] += self.lr * (reward + self.gamma * np.max(self.Q[state_plus,:]) - self.Q[state, action]) 
                
                state = state_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                if done or truncated:
                    break

                cum_step += 1

                # Schedule learning rate decay
                if cum_step >= self.step_start_decay_lr:
                    self.lr = self.lr_init / (cum_step - self.step_start_decay_lr + 1)
            
            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

    def Robust_Q_learning(self, n_episodes):
        """Robust Q-learning with the R-contamination Model"""
        
        cum_step = 0

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")
            
            state, info = self.env.reset() # starting state at 0

            while True:
                action = self.epsilon_greedy_policy(state)

                state_plus, reward, done, truncated, info = self.env.step(action)

                V = np.max(self.Q, axis = 1) # state-value function for all states
                # Worst-case transition
                # TODO: the following works only for the 4-by-4 frozenlake
                V_min = 1.0
                if state >= 4:
                    if V[state-4] < V_min:
                        V_min = V[state-4]
                        #action = 3
                if state < 16-4:
                    if V[state+4] < V_min:
                        V_min = V[state+4]
                        #action = 1
                if state % 4 > 0:
                    if V[state-1] < V_min:
                        V_min = V[state-1]
                        #action = 0
                if state % 4 < 3:
                    if V[state+1] < V_min:
                        V_min = V[state+1]
                        #action = 2
                target  = reward + self.gamma * self.R * V_min + self.gamma * (1 - self.R) * V[state_plus]
                #target  = reward + self.gamma * self.R * np.min(V) + self.gamma * (1 - self.R) * V[state_plus]
                
                self.Q[state, action] += self.lr * (target - self.Q[state, action]) 
                
                state = state_plus

                self.evaluation_return.append(np.max(self.Q[0, :]))

                if done or truncated:
                    break

                cum_step += 1

                # Schedule learning rate decay
                if cum_step >= self.step_start_decay_lr:
                    self.lr = self.lr_init / (cum_step - self.step_start_decay_lr + 1)

            # Schedule epsilon decay to impose GLIE
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

    def sim_perturbed(self, p):
        # p: with probability p, the transition is uniformly over S given (s,a); 
        #    with probability 1-p, the transition is the true transition.

        # Initialize the environment and state
        state, info = self.env.reset() # starting state at 0

        #state_hist = []
        G = 0
        I = 1
        V = np.max(self.Q, axis = 1)
        while True:
            
            if np.random.rand() <= p:
                #action = np.random.choice(self.n_action)
                # TODO: worst-case transition for 4-by-4 frozen lake
                V_min = 1.0
                if state >= 4:
                    if V[state-4] < V_min:
                        V_min = V[state-4]
                        action = 3
                if state < 16-4:
                    if V[state+4] < V_min:
                        V_min = V[state+4]
                        action = 1
                if state % 4 > 0:
                    if V[state-1] < V_min:
                        V_min = V[state-1]
                        action = 0
                if state % 4 < 3:
                    if V[state+1] < V_min:
                        V_min = V[state+1]
                        action = 2
            else:
                # The transition follows the greedy policy with probability 1-p
                action = np.argmax(self.Q[state, :])
            
            state_plus, reward, done, truncated, info = self.env.step(action)

            G += I * reward
            I = I * self.gamma

            # Move to the next state
            state = state_plus
                
            if done or truncated:
                break
        
        return G