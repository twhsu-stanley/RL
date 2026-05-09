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
        R: float = 0.0,
        C: float = 1.0,
        n_uncertainty_samples: int = 64
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
            self.observation_low = None
            self.observation_high = None
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.is_state_discrete = False
            self.dim_state = env.observation_space.shape[0]       # e.g. CartPole, Pendulum
            self.observation_low = torch.tensor(env.observation_space.low, dtype=torch.float32)
            self.observation_high = torch.tensor(env.observation_space.high, dtype=torch.float32)
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

        # Parameters for robust DQN with the localized R-C uncertainty set.
        self.R = R
        self.C = C
        self.n_uncertainty_samples = n_uncertainty_samples
        self.state_coordinates = self._build_state_coordinates() if self.is_state_discrete else None

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

    def convert_to_one_hot(self, x):
        x = nn.functional.one_hot(x.long().squeeze(), num_classes=self.dim_state).float()
        return x

    def _build_state_coordinates(self):
        """Build grid coordinates for discrete states, used by the R-C uncertainty set."""
        if hasattr(self.env.unwrapped, "desc"):
            n_rows, n_cols = self.env.unwrapped.desc.shape
            return np.array(
                [[s // n_cols, s % n_cols] for s in range(self.dim_state)],
                dtype=float
            )

        grid_size = int(np.sqrt(self.dim_state))
        if grid_size * grid_size == self.dim_state:
            return np.array(
                [[s // grid_size, s % grid_size] for s in range(self.dim_state)],
                dtype=float
            )

        return np.arange(self.dim_state, dtype=float).reshape(-1, 1)

    def _localized_uncertainty_indices(self, nominal_next_state_idx, C=None):
        """
        Discrete-state localized uncertainty set:
            C(s,a) = {s_tilde: ||s' - s_tilde|| <= C},
        where s' is the observed deterministic nominal next state.
        """
        C = self.C if C is None else C
        distances = np.linalg.norm(
            self.state_coordinates - self.state_coordinates[int(nominal_next_state_idx)],
            axis=1
        )
        return np.where(distances <= C + 1e-12)[0]

    def _sample_continuous_uncertainty_states(self, nominal_next_state, C=None):
        """
        Approximate the continuous-state localized uncertainty set by sampling states
        inside an L2 ball of radius C around the observed nominal next state s'.
        The nominal next state itself is always included as a candidate.
        """
        C = self.C if C is None else C
        center = nominal_next_state.detach().float().reshape(-1)

        if C <= 0 or self.n_uncertainty_samples <= 0:
            return center.unsqueeze(0)

        dim = center.numel()
        directions = torch.randn(self.n_uncertainty_samples, dim, dtype=torch.float32)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        radii = torch.rand(self.n_uncertainty_samples, 1, dtype=torch.float32).pow(1.0 / dim) * C
        samples = center.unsqueeze(0) + radii * directions
        samples = torch.cat([center.unsqueeze(0), samples], dim=0)

        # Respect finite observation-space bounds when they exist. CartPole has
        # infinite velocity bounds, so only finite coordinates are clamped.
        low = self.observation_low
        high = self.observation_high
        if low is not None and high is not None:
            finite_low = torch.isfinite(low)
            finite_high = torch.isfinite(high)
            if finite_low.any():
                samples[:, finite_low] = torch.maximum(samples[:, finite_low], low[finite_low])
            if finite_high.any():
                samples[:, finite_high] = torch.minimum(samples[:, finite_high], high[finite_high])

        return samples

    def _localized_uncertainty_state_candidates(self, nominal_next_state, C=None):
        """Return candidate states in the R-C uncertainty set centered at s'."""
        if self.is_state_discrete:
            nominal_next_state_idx = int(torch.argmax(nominal_next_state).item())
            uncertainty_indices = self._localized_uncertainty_indices(nominal_next_state_idx, C)
            uncertainty_indices = torch.tensor(uncertainty_indices, dtype=torch.long)
            return nn.functional.one_hot(uncertainty_indices, num_classes=self.dim_state).float()

        return self._sample_continuous_uncertainty_states(nominal_next_state, C)

    def _worst_case_values_R_C(self, next_state_batch, C=None):
        """
        Compute min_{s_tilde in C(s,a)} max_a Q_target(s_tilde, a)
        for every observed nonterminal nominal next state in next_state_batch.
        """
        worst_values = []
        for nominal_next_state in next_state_batch:
            candidate_states = self._localized_uncertainty_state_candidates(nominal_next_state, C)
            candidate_values = self.Q_net_target(candidate_states).max(1).values
            worst_values.append(candidate_values.min())

        if len(worst_values) == 0:
            return torch.empty((0, 1), dtype=torch.float32)

        return torch.stack(worst_values).unsqueeze(1)

    def _state_tensor_to_env_state(self, state_tensor):
        if self.is_state_discrete:
            return int(torch.argmax(state_tensor).item())
        return state_tensor.detach().cpu().numpy().astype(np.float64)

    def _set_env_state(self, state_tensor):
        """Overwrite the simulator internal state after an R-C perturbation."""
        if self.is_state_discrete:
            self.env.unwrapped.s = self._state_tensor_to_env_state(state_tensor)
        else:
            self.env.unwrapped.state = self._state_tensor_to_env_state(state_tensor)

    def _is_terminal_state(self, state_tensor):
        """Best-effort terminal check after manually perturbing the next state."""
        if self.is_state_discrete and hasattr(self.env.unwrapped, "desc"):
            state_idx = self._state_tensor_to_env_state(state_tensor)
            n_cols = self.env.unwrapped.desc.shape[1]
            row, col = divmod(state_idx, n_cols)
            cell = self.env.unwrapped.desc[row, col].decode("utf-8")
            return cell in ["H", "G"]

        # CartPole-v1 termination condition.
        env_id = self.env.spec.id if self.env.spec is not None else ""
        if (not self.is_state_discrete) and "CartPole" in env_id:
            x, x_dot, theta, theta_dot = self._state_tensor_to_env_state(state_tensor)
            x_threshold = self.env.unwrapped.x_threshold
            theta_threshold = self.env.unwrapped.theta_threshold_radians
            return bool(
                x < -x_threshold
                or x > x_threshold
                or theta < -theta_threshold
                or theta > theta_threshold
            )

        return False

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

        # R-C robust target:
        #   r + gamma*(1-R)*V(s') + gamma*R*min_{s_tilde in C(s,a)} V(s_tilde),
        # where V(s) = max_a Q_{theta-}(s,a), and C(s,a) is the radius-C
        # ball centered at the observed deterministic nominal next state s'.
        V_plus = torch.zeros((self.batch_size, 1))
        V_worst = torch.zeros((self.batch_size, 1))
        with torch.no_grad():
            if next_state_batch.shape[0] > 0:
                V_plus[not_none_mask] = self.Q_net_target(next_state_batch).max(1).values.unsqueeze(1)
                V_worst[not_none_mask] = self._worst_case_values_R_C(next_state_batch)
        Q_target = reward_batch + self.gamma * (1 - self.R) * V_plus + self.gamma * self.R * V_worst

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
    
    def DQN_sim_perturbed(self, p, C=None):
        # p: with probability p, the transition is perturbed inside the
        # localized radius-C uncertainty set centered at the nominal next state;
        # with probability 1-p, the transition is the nominal simulator transition.

        C = self.C if C is None else C

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

        G = 0
        I = 1
        while True:
            # Select greedy action under the learned Q net.
            action = self.epsilon_greedy_policy(state)
            state_plus, reward, done, truncated, info = self.env.step(action.item())

            if done or truncated:
                state_plus_tensor = None
            else:
                if self.is_state_discrete:
                    state_plus_tensor = torch.tensor(state_plus, dtype = torch.float32).unsqueeze(0)
                    state_plus_tensor = self.convert_to_one_hot(state_plus_tensor)
                else:
                    state_plus_tensor = torch.tensor(state_plus, dtype = torch.float32)

                # Apply the R-C perturbation to the next state with probability p.
                if np.random.rand() <= p:
                    with torch.no_grad():
                        candidate_states = self._localized_uncertainty_state_candidates(state_plus_tensor, C)
                        candidate_values = self.Q_net(candidate_states).max(1).values
                        worst_idx = candidate_values.argmin().item()
                        state_plus_tensor = candidate_states[worst_idx]

                    self._set_env_state(state_plus_tensor)
                    done = self._is_terminal_state(state_plus_tensor)

                    if self.is_state_discrete and hasattr(self.env.unwrapped, "desc"):
                        state_idx = self._state_tensor_to_env_state(state_plus_tensor)
                        n_cols = self.env.unwrapped.desc.shape[1]
                        row, col = divmod(state_idx, n_cols)
                        cell = self.env.unwrapped.desc[row, col].decode("utf-8")
                        reward = 1.0 if cell == "G" else 0.0

            G += I * reward
            I = I * self.gamma

            # Store data
            state_hist.append(state)

            if done or truncated:
                break
            
            # Move to the next state
            state = state_plus_tensor
        
        return G, state_hist
