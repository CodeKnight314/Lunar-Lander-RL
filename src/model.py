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
    
class ContinuousQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int): 
        super(ContinuousQN, self).__init__()
        self.net = nn.Sequential(*[
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  
            nn.ReLU(),
            nn.Linear(128, output_dim), 
            nn.Tanh()
        ])
        
    def forward(self, x):
        return self.net(x)
    
    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
class Critic(nn.Module): 
    def __init__(self, input_dim: int, output_dim: int): 
        super(Critic, self).__init__()
        self.net = nn.Sequential(*[
            nn.Linear(input_dim + output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  
            nn.ReLU(),
            nn.Linear(128, 1)
        ])
        
    def forward(self, x, action):
        x = torch.cat([x, action], dim=1)
        return self.net(x)
    
    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)