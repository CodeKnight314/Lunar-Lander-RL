import cv2 
import gym
from src.model import ContinuousQN, Critic
from src.replay import ReplayBuffer

import torch 
import torch.nn as nn 
import torch.optim as optim
import yaml
import os
from tqdm import tqdm

class C_environment:
    def __init__(self, config_path: str, weights: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.env = gym.make("LunarLanderContinuous-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        