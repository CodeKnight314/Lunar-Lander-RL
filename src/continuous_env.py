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
import random 

class C_environment:
    def __init__(self, config_path: str, actor_weights: str = None, critic_weights: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        
        self.env = gym.make("LunarLanderContinuous-v2", render_mode="rgb_array")
        self.actor = ContinuousQN(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(self.device)
        self.critic = Critic(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(self.device)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.config["critic_lr"])
        
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(self.config["max_memory"])
        
        self.epsilon = self.config["epsilon"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        
        if actor_weights: 
            self.actor.load(actor_weights)
        if critic_weights: 
            self.critic.load(critic_weights)
            
            
    def train_continuous(self, path: str = None):
        avg_awards = []
        avg_grad = [0]
        avg_loss = [0]
        
        pbar = tqdm(range(self.config["episode"]))
        steps = 0
        for epsiode in pbar: 
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done: 
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else: 
                    with torch.no_grad():
                        action = self.actor(torch.tensor(state).float().to(self.device))
                        action = action.squeeze(0).numpy()
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = truncated or terminated
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(self.replay_buffer) > self.config["batch_size"]:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config["batch_size"])
                    with torch.no_grad(): 
                        next_actions = self.actor(states)
                        target_q = rewards + self.config["gamma"] * self.critic(next_states, next_actions) * (1 - dones)
                        
                    current_q = self.critic(states, actions)
                    critic_loss = self.criterion(current_q, target_q)
                    
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    
                    grad_norm = 0.0 
                    for name, param in self.critic.named_parameters(): 
                        if param.grad is not None: 
                            grad_norm += param.grad.norm().item() 
                    
                    avg_grad.append(grad_norm)
                    self.critic_optim.step()
                    avg_loss.append(critic_loss.item())
                    
                    actor_actions = self.actor(states)
                    actor_loss = -self.critic(states, actor_actions).mean()

                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    grad_norm = 0.0
                    for name, param in self.actor.named_parameters(): 
                        if param.grad is not None: 
                            grad_norm += param.grad.norm().item()
                    avg_grad.append(grad_norm)
                    self.actor_optim.step()
                    
                    avg_loss.append(actor_loss.item())
                    
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            avg_awards.append(total_reward)
            
            pbar.postfix = f"Avg Reward: {sum(avg_awards) / len(avg_awards):.4f}, Epsilon: {self.epsilon:.4f}, Loss: {sum(avg_loss)/len(avg_loss):.4f}, Grad: {sum(avg_grad)/len(avg_grad):.4f}"
        
        self.actor.save(os.path.join(path, "c_lander.pth"))
        self.critic.save(os.path.join(path, "c_critic.pth"))
            
    def test_dqn(self, path: str):
        self.actor.load(os.path.join(path, "c_lander.pth"))
        
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
                action = self.actor(torch.tensor(state).float().to(self.device))
                action = action.squeeze(0).cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if step_count % 20 == 0:
                print(f"Recording step {step_count}, current reward: {total_reward}")
        
        video.release()
        print(f"MP4 video saved to {video_path}")
        print(f"Test completed with total reward: {total_reward}")
        return total_reward
        
    def close(self):
        self.env.close()
                    