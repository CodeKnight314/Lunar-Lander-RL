import random
import torch 
import numpy as np 

class ReplayBuffer: 
    def __init__(self, max_memory: int):
        self.buffer = []
        self.max_memory = max_memory
        
    def add(self, state, action, reward, next_state, done):
        while(len(self.buffer) >= self.max_memory):
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)