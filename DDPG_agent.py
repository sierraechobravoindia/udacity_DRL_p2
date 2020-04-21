from models import Actor, Critic

import random
import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple

GAMMA = 0.99
TAU = 1e-3
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128

LR_ACTOR = 2e-4
LR_CRITIC = 2e-4

WEIGHT_DECAY = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    """Defines the buffer to store experiences"""
    
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size= batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
                                     
        return (states, actions, rewards, next_states, dones)
                                     
    def __len__(self):
        return len(self.memory)
                                     
                                     
                                     
class Agent():
    """Defines the agent that interacts with the environment"""
    
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
                                     
        self.actor = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
                                     
        self.critic = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = LR_CRITIC)
        
        self.copy_init_weights(self.actor, self. actor_target)
        self.copy_init_weights(self.critic, self.critic_target)
        
        self.noise = OUNoise(action_size, seed)
            
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)                    
    
    def copy_init_weights(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
    
    
    
    def act(self, state, exploration= True):
        
        state = torch.from_numpy(state).float().to(device)                     
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()  
                         
        if exploration:
            action += self.noise.sample()  
            
        return np.clip(action, -1, 1)
                                     
                                     
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
    
    def learn(self, mini_batch, gamma):
        states, actions, rewards, next_states, dones = mini_batch
        
        #train actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()                            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()                             
                                     
        #train critic
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)                             
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()                 
                                     
        # soft update of target network
        self.soft_update(self.actor, self.actor_target, TAU)                             
        self.soft_update(self.critic, self.critic_target, TAU)                   
                                     
                                     
    def soft_update(self, original, target, tau):
        for target_param, param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
                                     
    
def Noise(action_size, seed):
   random.seed(seed)
   mu, sigma = 0, 0.2 
   noise = np.random.normal(mu, sigma, action_size) 
   return np.array(noise)


class OUNoise():
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
        
      







