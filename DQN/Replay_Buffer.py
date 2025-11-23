import random
import numpy as np
from collections import deque

import torch

class Replay_Buffer:
    """A deque object that stores the (s, a, s', r) tuples"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.deque = deque([], maxlen=capacity)

    def append(self, state, action, state_plus, reward):
        """ Append a new (s, a, s', r) instance to the replay buffer"""
        # Note: If the deque reaches maxlen and a new element is appended,
        #       the element at the opposite end is automatically removed.
        self.deque.append((state, action, state_plus, reward))

    def sample(self, batch_size):
        batch = random.sample(self.deque, batch_size)
        return batch
    
    def length(self):
        return len(self.deque)