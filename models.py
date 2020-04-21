import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def init_interval(layer):
    size_in = layer.weight.data.size()[0]
    limit = 1./np.sqrt(size_in)
    return (-limit, limit)

class Actor(nn.Module):
    """Defines the policy model"""
    
    def __init__(self, state_size, action_size, seed, fc1_size=128, fc2_size=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.bn2 = nn.BatchNorm1d(fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn3 = nn.BatchNorm1d(fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)
        
        #initialize the weights
        self.fc1.weight.data.uniform_(*init_interval(self.fc1))
        self.fc2.weight.data.uniform_(*init_interval(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self, state):
        if state.dim()==1:
            state = state.unsqueeze(0)
        
        x = F.relu(self.fc1(state))
        x = self.bn2(x) 
        x = F.relu(self.fc2(x))
    
        return F.tanh(self.fc3(x))
    
    
class Critic(nn.Module):
    """Defines the Value model"""
    
    def __init__(self, state_size, action_size, seed, fc1_size=128, fc2_size=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size+action_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        
        #initialize weights
        self.fc1.weight.data.uniform_(*init_interval(self.fc1))
        self.fc2.weight.data.uniform_(*init_interval(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self, state, action):
        if state.dim() == 1:
            state.unsqueeze(0)
            
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = torch.cat([x, action], 1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)