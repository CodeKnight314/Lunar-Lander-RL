import torch 
import torch.nn as nn 
import numpy as np

def init_weights(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)
    return module

class DiscreteQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DiscreteQN, self).__init__()
        self.net = nn.Sequential(*[
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ])
        
    def forward(self, x):
        return self.net(x)
    
    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
class DuelDQN(nn.Module): 
    def __init__(self, input_dim: int, output_dim: int): 
        super().__init__()
        self.main = nn.Sequential(*[
            nn.Linear(input_dim, 128), 
            nn.ReLU()
        ])
        
        self.value = nn.Sequential(*[
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, 1)
        ])
        
        self.advantage = nn.Sequential(*[
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ])
        
    def forward(self, x):
        x = self.main(x)
        value = self.value(x)
        advantage = self.advantage(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
class ActorContinuous(nn.Module):
    def __init__(self, input_dim: int, output_dim: int): 
        super(ActorContinuous, self).__init__()
        self.net = nn.Sequential(*[
            init_weights(nn.Linear(input_dim, 256)),
            nn.ReLU(),
            init_weights(nn.Linear(256, 256)),  
            nn.ReLU()
        ])
        
        self.mean = init_weights(nn.Linear(256, output_dim))
        
        self.log_std = nn.Parameter(torch.ones(output_dim) * -0.5)
        
    def forward(self, x):
        features = self.net(x)
        mean = torch.tanh(self.mean(features))
        
        log_std = torch.clamp(self.log_std, -20, 20)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, x): 
        mean, std = self.forward(x)
        
        if torch.isnan(mean).any() or torch.isnan(std).any():
            if torch.isnan(mean).any():
                mean = torch.zeros_like(mean)
            if torch.isnan(std).any():
                std = torch.ones_like(std) * 0.1
                
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, -1, 1)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob = torch.clamp(log_prob, -20, 20)
        
        return action, log_prob
    
    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
class Critic(nn.Module): 
    def __init__(self, input_dim: int): 
        super(Critic, self).__init__()
        self.net = nn.Sequential(*[
            init_weights(nn.Linear(input_dim, 256)),
            nn.ReLU(),
            init_weights(nn.Linear(256, 256)),  
            nn.ReLU(),
            init_weights(nn.Linear(256, 1))
        ])
        
    def forward(self, x):
        return self.net(x)
    
    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)