import torch 
import torch.nn as nn 

class DiscreteQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DiscreteQN, self).__init__()
        self.net = nn.Sequential(*[
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  
            nn.ReLU(),
            nn.Linear(256, output_dim)
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
        advantage = self.value(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
class ActorContinuous(nn.Module):
    def __init__(self, input_dim: int, output_dim: int): 
        super(ActorContinuous, self).__init__()
        self.net = nn.Sequential(*[
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  
            nn.ReLU()
        ])
        
        self.mean = nn.Linear(128, output_dim)
        
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        features = self.net(x)
        mean = torch.tanh(self.mean(features))
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample(self, x): 
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
class Critic(nn.Module): 
    def __init__(self, input_dim: int): 
        super(Critic, self).__init__()
        self.net = nn.Sequential(*[
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  
            nn.ReLU(),
            nn.Linear(128, 1)
        ])
        
    def forward(self, x):
        return self.net(x)
    
    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)