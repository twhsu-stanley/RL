import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from Policy_Net import Policy_Net
from V_Net import V_Net

class PG_Agent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float,
        learning_rate_policy: float,
        learning_rate_value: float = None
    ):
        self.env = env
        self.gamma = gamma
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_value = learning_rate_value

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

        self.policy_net = Policy_Net(self.dim_state, self.dim_action)
        self.value_net = V_Net(self.dim_state) # for actor-critic algorithms

        self.optimizer_policy = optim.SGD(self.policy_net.parameters(), lr = self.learning_rate_policy)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr = self.learning_rate_value) if self.learning_rate_value else None

        self.evaluation_return = []
    
    def convert_to_one_hot(self, x):
        x = torch.nn.functional.one_hot(x.long().squeeze(), num_classes=self.dim_state).float()
        return x
    
    def REINFORCE_learning(self, n_episodes):
        """ The REINFORCE algorithm"""

        for episode in range(n_episodes):
            # Initialize the environment and state
            state, info = self.env.reset() # starting state at 0
            
            # Convert to one-hot vector if discrete state space
            if self.is_state_discrete:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # device=device
                state = self.convert_to_one_hot(state)
            else:
                state = torch.tensor(state, dtype=torch.float32) # device=device

            # Simulate a complete episode
            state_hist = []
            action_hist = []
            reward_hist = []
            while True:
                action = self.policy_net.get_action(state)

                state_plus, reward, done, truncated, info = self.env.step(action.item())

                if self.is_state_discrete:
                    state_plus = torch.tensor(state_plus, dtype = torch.float32).unsqueeze(0) # device=device
                    state_plus = self.convert_to_one_hot(state_plus)
                else:
                    state_plus = torch.tensor(state_plus, dtype = torch.float32) # device=device

                # Store state, action, reward
                state_hist.append(state)
                action_hist.append(action)
                reward_hist.append(reward)

                # Move to the next state
                state = state_plus

                if done or truncated:
                    break
            
            for t in reversed(range(len(state_hist))):
                G_t = 0
                for k in range(t, len(state_hist)):
                    G_t = G_t + (self.gamma**(k - t)) * reward_hist[k]
                if t == 0:
                    G_0 = G_t
                    self.evaluation_return.append(G_0)
                    print(f"Episode {episode+1}/{n_episodes} : G_0 = {G_0}")
                
                state_t = state_hist[t]
                action_t = action_hist[t]
                loss = -self.gamma**t * G_t * self.policy_net.get_distribution(state_t, with_grad=True).log_prob(action_t)

                self.optimizer_policy.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 3) # gradient clipping
                self.optimizer_policy.step() # single step of SGD

    def ACTOR_CRITIC_learning(self, n_episodes):
        """ The Actor-Critic algorithm"""
        
        for episode in range(n_episodes):
            # Initialize the environment and state
            state, info = self.env.reset() # starting state at 0
            
            
            # Convert to one-hot vector if discrete state space
            if self.is_state_discrete:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # device=device
                state = self.convert_to_one_hot(state)
            else:
                state = torch.tensor(state, dtype=torch.float32) # device=device
            
            with torch.no_grad():
                self.state_init = state.clone()

            I = 1
            G_0 = 0
            while True:
                action = self.policy_net.get_action(state)

                state_plus, reward, done, truncated, info = self.env.step(action.item())

                if self.is_state_discrete:
                    state_plus = torch.tensor(state_plus, dtype = torch.float32).unsqueeze(0) # device=device
                    state_plus = self.convert_to_one_hot(state_plus)
                else:
                    state_plus = torch.tensor(state_plus, dtype = torch.float32) # device=device
                
                G_0 += reward * I

                if done:
                    advantage = reward - self.value_net(state)
                else:
                    advantage = reward + self.gamma * self.value_net(state_plus).detach() - self.value_net(state)
                    # NOTE: detach the bootstrap target so policy update does not
                    #       try to backprop through the value network a second time

                # Critic (value net) update
                #loss_value = -advantage.detach().item() * self.value_net(state)
                loss_value = advantage.pow(2)
                self.optimizer_value.zero_grad()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 10) # gradient clipping
                self.optimizer_value.step() # single step of SGD

                # Actor (policy net) update
                loss_policy = -I * advantage.detach().item() * self.policy_net.get_distribution(state, with_grad=True).log_prob(action)
                self.optimizer_policy.zero_grad()
                loss_policy.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 3) # gradient clipping
                self.optimizer_policy.step() # single step of SGD
                I = I * self.gamma

                # Move to the next state
                state = state_plus

                if done or truncated:
                    self.evaluation_return.append(G_0)
                    print(f"Episode {episode+1}/{n_episodes} : G_0 = {G_0}")
                    break
