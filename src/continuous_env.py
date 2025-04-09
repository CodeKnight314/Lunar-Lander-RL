import cv2 
import gym
import torch.distributions
from model import ActorContinuous, Critic
from replay import ReplayBuffer

import torch 
import torch.nn as nn 
import torch.optim as optim
import yaml
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class C_environment:
    def __init__(self, config_path: str, actor_weights: str = None, critic_weights: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        
        self.env = gym.make("LunarLanderContinuous-v2", render_mode="rgb_array")
        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.actor = ActorContinuous(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim + action_dim).to(self.device)
        
        self.target_actor = ActorContinuous(obs_dim, action_dim).to(self.device)
        self.target_critic = Critic(obs_dim + action_dim).to(self.device)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.config["critic_lr"])
        
        self.buffer = ReplayBuffer(self.config["max_memory"])
        
        self.criterion = nn.MSELoss()
        
        self.target_update_freq = self.config['target_update_steps']
        
        if actor_weights: 
            self.actor.load(actor_weights)
        if critic_weights: 
            self.critic.load(critic_weights)
    
    def update_target_network(self): 
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
    def train_continuous(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        avg_reward = [] 
        avg_actor_loss = []
        avg_critic_loss = []
        
        pbar = tqdm(range(self.config["episode"]))
        steps = 0
        for episode in pbar: 
            state, _ = self.env.reset() 
            total_reward = 0
            done = False
            
            while not done: 
                action = self.actor.sample(torch.tensor(state).float().to(self.device))
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state 
                total_reward += reward
                
                if len(self.buffer) > self.config["batch_size"]:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.config["batch_size"])
                    
                    with torch.no_grad(): 
                        next_actions, _ = self.target_actor.sample(states)
                        q_target_next = self.target_critic(torch.cat([next_states, next_actions], dim=1))
                        q_target = rewards + self.config["gamma"] * (1 - dones) * q_target_next
                        
                    q_pred = self.critic(torch.cat([states, actions], dim=1))
                    critic_loss = self.criterion(q_pred, q_target)
                    
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
                    
                    curr_actions, _ = self.actor.sample(states)
                    actor_loss = -self.critic(torch.cat([states, curr_actions], dim=1)).mean()

                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                    
                    avg_actor_loss.append(actor_loss.item())
                    avg_critic_loss.append(critic_loss.item())

                steps += 1
                if steps % self.target_update_freq == 0:
                    self.update_target_network()

            avg_reward.append(total_reward)

            pbar.set_postfix({
                "Avg Reward": np.mean(avg_reward) if len(avg_reward) > 0 else 0,
                "Actor Loss": np.mean(avg_actor_loss) if len(avg_critic_loss) > 0 else 0,
                "Critic Loss": np.mean(avg_critic_loss) if len(avg_critic_loss) > 0 else 0
            })
            
        self.critic.save(os.path.join(path, "c_critic.pth"))
        self.actor.save(os.path.join(path, "c_actor.pth"))
    
    def test_continuous(self, path: str):
        os.makedirs(path, exist_ok=True)

        self.actor.load(os.path.join(path, "c_actor.pth"))
        video_path = os.path.join(path, "lander_video.mp4")
        
        state, _ = self.env.reset()
        
        total_reward = 0
        done = False 
        
        frame = self.env.render()
        height, width, layers = frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        frames = []
        
        total_reward = 0
        done = False 
        step_count = 0
        
        while not done: 
            frame = self.env.render() 
            frames.append(frame)
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
            
            with torch.no_grad(): 
                action, _ = self.actor.sample(state)
                
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            doen = terminated or truncated 
            
            state = next_state 
            total_reward += reward 
            step_coiunt += 1 
            
            if step_count % 20 == 0: 
                print(f"Recording step {step_count}, current reward: {total_reward}")
                
        video.release()
        print(f"MP4 video saved to {video_path}")
        print(f"Test completed with total reward: {total_reward}")
        return total_reward
    
    def close(self): 
        self.env.close()