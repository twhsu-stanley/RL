import torch
import torch.nn as nn
import torch.nn.functional as F

class V_Net(nn.Module):
    """Value function network for Actor-Critic"""

    def __init__(self, dim_states):
        super().__init__()
        self.layer1 = nn.Linear(dim_states, 12)
        self.layer2 = nn.Linear(12, 8)
        self.layer3 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x