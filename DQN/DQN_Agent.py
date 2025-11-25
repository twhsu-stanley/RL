import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from Q_Net import Q_Net
from Replay_Buffer import Replay_Buffer

class DQN_Agent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float,
        learning_rate_init: float,
        epsilon_init: float,
        epsilon_lb: float = 0.01,
        epsilon_decay_rate: float = 0.999,
        batch_size: int = 32,
        replay_buffer_capacity: int = 4000,
        Q_net_target_update_freq: int = 10,
        R: float = 0.0
    ):
        self.env = env
        self.gamma = gamma
        self.learning_rate_init = learning_rate_init
        self.epsilon_init = epsilon_init
        self.epsilon = self.epsilon_init
        self.epsilon_lb = epsilon_lb
        self.epsilon_decay_rate = epsilon_decay_rate

        self.cumulative_steps = 0

        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.is_state_discrete = True
            self.dim_state = env.observation_space.n              # e.g. FrozenLake
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.is_state_discrete = False
            self.dim_state = env.observation_space.shape[0]       # e.g. CartPole, Pendulum
        else:
            raise NotImplementedError(f"Unsupported observation space type: {type(env.observation_space)}")

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.is_action_discrete = True
            self.dim_action = env.action_space.n              # e.g. FrozenLake, CartPole
        #elif isinstance(env.action_space, gym.spaces.Box):
        #    self.is_action_discrete = False
        #    self.dim_action = env.action_space.shape[0]       # e.g. Pendulum
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(env.action_space)}")

        self.Q_net = Q_Net(self.dim_state, self.dim_action)
        self.Q_net_target = Q_Net(self.dim_state, self.dim_action)
        self.Q_net_target.load_state_dict(self.Q_net.state_dict())

        self.Q_net_target_update_freq = Q_net_target_update_freq  # update target network every 50 steps
        self.batch_size = batch_size
        self.replay_buffer = Replay_Buffer(replay_buffer_capacity)

        self.optimizer = optim.Adam(self.Q_net.parameters(), lr = self.learning_rate_init)
        #self.optimizer = optim.SGD(self.Q_net.parameters(), lr = self.learning_rate_init)

        # Initialize the evaluation return V(x_0)
        state_init, info = self.env.reset() # starting state at 0
        # Convert to one-hot vector if discrete state space
        if self.is_state_discrete:
            self.state_init = torch.tensor(state_init, dtype = torch.float32).unsqueeze(0) # device=device
            self.state_init = self.convert_to_one_hot(self.state_init)
        else: 
            self.state_init = torch.tensor(state_init, dtype = torch.float32)
            
        self.evaluation_return = []
        with torch.no_grad():
            self.evaluation_return.append(self.Q_net(self.state_init).max(0).values.item())

        # Parameters for robust DQN
        self.R = R
        if R > 0:
            if self.is_state_discrete:
                self.all_states = self.convert_to_one_hot(torch.tensor(np.arange(self.dim_state)))
            else:
                raise NotImplementedError("Robust DQN currently doesn't work for continuous state space")

    def convert_to_one_hot(self, x):
        x = nn.functional.one_hot(x.long().squeeze(), num_classes=self.dim_state).float()
        return x

    def epsilon_greedy_policy(self, state):
        """ Epsilon-greedy policy based on Q_net"""

        if np.random.rand() <= self.epsilon:
            # random exploration with probability epsilon
            # TODO: consider continuous action space
            action = torch.tensor([np.random.choice(self.dim_action)]) #, device=device)? 
        else:
            # greedy action with probability 1-epsilon
            with torch.no_grad():
                action = self.Q_net(state).max(0).indices.unsqueeze(0)  #, device=device)? 
        return action

    def DQN_SGD_step(self):
        """ Performs a single-step SGD update of the DQN parameters (theta)"""
        
        # Sample a batch of (s, a, s', r) from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        #if self.is_state_discrete:
        state_batch = torch.stack([s for (s, a, s_plus, r) in batch])
        #else:
        #    state_batch = torch.cat([s for (s, a, s_plus, r) in batch]).unsqueeze(1)
        action_batch = torch.cat([a for (s, a, s_plus, r) in batch]).unsqueeze(1)
        reward_batch = torch.cat([r for (s, a, s_plus, r) in batch]).unsqueeze(1)

        # Handle next_state being None (terminal state)
        not_none_mask = torch.tensor([s_plus is not None for (s, a, s_plus, r) in batch], dtype=torch.bool)
        if not_none_mask.sum().item() > 0:
            #if self.is_state_discrete:
            next_state_batch = torch.stack([s_plus for (s, a, s_plus, r) in batch if s_plus is not None])
            #else:
                #next_state_batch = torch.cat([s_plus for (s, a, s_plus, r) in batch if s_plus is not None]).unsqueeze(1)        
        else:
            next_state_batch = torch.empty((0, state_batch.shape[1]))

        # Compute Q_{\theta}(s,a)
        Q = self.Q_net(state_batch).gather(1, action_batch)

        # Compute r + \gamma * max_a' Q_{\theta-}(s',a')
        V_plus = torch.zeros((self.batch_size, 1))
        with torch.no_grad():
            if next_state_batch.shape[0] > 0:
                V_plus[not_none_mask] = self.Q_net_target(next_state_batch).max(1).values.unsqueeze(1)
        Q_target = reward_batch + self.gamma * V_plus

        # Compute the 2-norm loss
        criterion = nn.MSELoss()
        #criterion = clampedL2Loss() # clip the loss between [-1, 1]?
        loss = criterion(Q, Q_target)

        # Compute the gradients and perform a single SGD step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_net.parameters(), 3) # gradient clipping
        self.optimizer.step()

        # Update the target network every self.Q_net_target_update_freq steps
        if self.cumulative_steps % self.Q_net_target_update_freq == 0:
            self.Q_net_target.load_state_dict(self.Q_net.state_dict())

        with torch.no_grad():
            self.evaluation_return.append(self.Q_net(self.state_init).max(0).values.item())

    def Robust_DQN_SGD_step(self):
        """ Performs a single-step SGD update of the Robust DQN parameters (theta)"""
        
        # Sample a batch of (s, a, s', r) from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        #if self.is_state_discrete:
        state_batch = torch.stack([s for (s, a, s_plus, r) in batch])
        #else:
        #    state_batch = torch.cat([s for (s, a, s_plus, r) in batch]).unsqueeze(1)
        action_batch = torch.cat([a for (s, a, s_plus, r) in batch]).unsqueeze(1)
        reward_batch = torch.cat([r for (s, a, s_plus, r) in batch]).unsqueeze(1)

        # Handle next_state being None (terminal state)
        not_none_mask = torch.tensor([s_plus is not None for (s, a, s_plus, r) in batch], dtype=torch.bool)
        if not_none_mask.sum().item() > 0:
            #if self.is_state_discrete:
            next_state_batch = torch.stack([s_plus for (s, a, s_plus, r) in batch if s_plus is not None])
            #else:
                #next_state_batch = torch.cat([s_plus for (s, a, s_plus, r) in batch if s_plus is not None]).unsqueeze(1)        
        else:
            next_state_batch = torch.empty((0, state_batch.shape[1]))

        # Compute Q_{\theta}(s,a)
        Q = self.Q_net(state_batch).gather(1, action_batch)

        # Compute r + \gamma * max_a' Q_{\theta-}(s',a')
        V_plus = torch.zeros((self.batch_size, 1))
        #V_min = 0
        with torch.no_grad():
            if next_state_batch.shape[0] > 0:
                V_plus[not_none_mask] = self.Q_net_target(next_state_batch).max(1).values.unsqueeze(1)
            # Find min V corresponding to the worst-case transition; currently only works for discrete state space
            V_min = torch.min(self.Q_net_target(self.all_states).max(1).values).item()
                #V_min = torch.min(self.Q_net_target(next_state_batch).max(1).values).item()
        Q_target = reward_batch + self.gamma * (1 - self.R) * V_plus + self.gamma * self.R * V_min

        # Compute the 2-norm loss
        criterion = nn.MSELoss()
        loss = criterion(Q, Q_target)

        # Compute the gradients and perform a single SGD step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_net.parameters(), 3) # gradient clipping
        self.optimizer.step()

        # Update the target network every self.Q_net_target_update_freq steps
        if self.cumulative_steps % self.Q_net_target_update_freq == 0:
            self.Q_net_target.load_state_dict(self.Q_net.state_dict())

        with torch.no_grad():
            self.evaluation_return.append(self.Q_net(self.state_init).max(0).values.item())

    def DQN_learning(self, n_episodes):
        """ The DQN algorithm"""

        for episode in range(n_episodes):
            print(f"Episode {episode+1}/{n_episodes}")
            
            # Initialize the environment and state
            state, info = self.env.reset() # starting state at 0
            
            # Convert to one-hot vector if discrete state space
            if self.is_state_discrete:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                state = self.convert_to_one_hot(state)
            else:
                state = torch.tensor(state, dtype=torch.float32)

            #I = 1
            #G_0 = 0
            while True:
                action = self.epsilon_greedy_policy(state)

                state_plus, reward, done, truncated, info = self.env.step(action.item())
                #G_0 += reward * I
                reward = torch.tensor(reward, dtype = torch.float32).unsqueeze(0)

                if done or truncated:
                    state_plus = None
                else:
                    if self.is_state_discrete:
                        state_plus = torch.tensor(state_plus, dtype = torch.float32).unsqueeze(0)
                        state_plus = self.convert_to_one_hot(state_plus)
                    else:
                        state_plus = torch.tensor(state_plus, dtype = torch.float32)

                # Store the transition in replay buffer
                self.replay_buffer.append(state, action, state_plus, reward)

                # Move to the next state
                state = state_plus

                # Single-step SGD update of the parameter
                if self.replay_buffer.length() == self.replay_buffer.capacity: #>= self.batch_size:
                    if self.R > 0:
                        self.Robust_DQN_SGD_step()
                    else:
                        self.DQN_SGD_step()
                    self.cumulative_steps += 1

                if done or truncated:
                    #self.evaluation_return.append(G_0)
                    break
            
            # Schedule epsilon decay
            self.epsilon = max(self.epsilon_lb, self.epsilon * self.epsilon_decay_rate)

    def DQN_sim(self):
        """Simulation using the learned Q net"""
       
        # Initialize the environment and state
        state, info = self.env.reset() # starting state at 0

        # Exploit the learned Q function
        self.epsilon = 0
            
        # Convert to one-hot vector if discrete state space
        if self.is_state_discrete:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state = self.convert_to_one_hot(state)
        else:
            state = torch.tensor(state, dtype=torch.float32)

        state_hist = [state]
        action_hist = []
        reward_hist = []
        while True:
            # Select action using epsilon-greedy policy given Q net
            action = self.epsilon_greedy_policy(state)

            state_plus, reward, done, truncated, info = self.env.step(action.item())
            reward = torch.tensor(reward, dtype = torch.float32).unsqueeze(0)

            if done or truncated:
                state_plus = None
            else:
                if self.is_state_discrete:
                    state_plus = torch.tensor(state_plus, dtype = torch.float32).unsqueeze(0)
                    state_plus = self.convert_to_one_hot(state_plus)
                else:
                    state_plus = torch.tensor(state_plus, dtype = torch.float32)

            # Store data
            state_hist.append(state)
            action_hist.append(action)
            reward_hist.append(reward)

            # Move to the next state
            state = state_plus
                
            if done or truncated:
                break
        
        return state_hist, action_hist, reward_hist
    
    def DQN_sim_perturbed(self, p):
        # p: with probability p, the transition is uniformly over S given (s,a); 
        #    with probability 1-p, the transition is the true transition.

        # Initialize the environment and state
        state, info = self.env.reset() # starting state at 0

        # Exploit the learned Q function
        self.epsilon = 0
            
        # Convert to one-hot vector if discrete state space
        if self.is_state_discrete:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state = self.convert_to_one_hot(state)
        else:
            state = torch.tensor(state, dtype=torch.float32)

        G = 0
        I = 1
        #V = np.max(self.Q, axis = 1)
        while True:
            if np.random.rand() <= p:
                action = np.random.choice(self.dim_action)
                state_plus, reward, done, truncated, info = self.env.step(action)
                # TODO: worst-case transition for 4-by-4 frozen lake
                """
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
                """
            else:
                # Select action using epsilon-greedy policy given Q net
                action = self.epsilon_greedy_policy(state)
                state_plus, reward, done, truncated, info = self.env.step(action.item())

            G += I * reward
            I = I * self.gamma

            if done or truncated:
                state_plus = None
            else:
                if self.is_state_discrete:
                    state_plus = torch.tensor(state_plus, dtype = torch.float32).unsqueeze(0)
                    state_plus = self.convert_to_one_hot(state_plus)
                else:
                    state_plus = torch.tensor(state_plus, dtype = torch.float32)
        
            # Move to the next state
            state = state_plus
                
            if done or truncated:
                break
        
        return G