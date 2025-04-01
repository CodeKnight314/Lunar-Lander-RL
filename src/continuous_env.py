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
        
        self.hard_update_target_networks()
        
        self.criterion = nn.MSELoss()
        
        if actor_weights: 
            self.actor.load(actor_weights)
        if critic_weights: 
            self.critic.load(critic_weights)
    
    def hard_update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update_target_networks(self, tau=0.005):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
    def train_continuous(self, path: str = None):
        avg_rewards = []
        actor_losses = []
        critic_losses = []
        
        pbar = tqdm(range(self.config["episode"]), desc="[Episode]")
        for episode in pbar:
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if len(self.buffer) < self.config["batch_size"]:
                    action = self.env.action_space.sample()
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        mean, std = self.actor(state_tensor)
                        action = torch.clamp(torch.distributions.Normal(mean, std).sample(), -1, 1)
                        action = action.cpu().numpy().squeeze(0)
                        
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    action = np.clip(action + noise, -1, 1)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                
                if len(self.buffer) >= self.config["batch_size"]:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.config["batch_size"])
                    states = states.to(self.device)
                    actions = actions.to(self.device).float()
                    rewards = rewards.to(self.device)
                    next_states = next_states.to(self.device)
                    dones = dones.to(self.device)
                    
                    with torch.no_grad():
                        next_means, next_stds = self.target_actor(next_states)
                        next_actions = torch.clamp(torch.distributions.Normal(next_means, next_stds).sample(), -1, 1)
                        next_q_values = self.target_critic(torch.cat([next_states, next_actions], dim=1))
                        target_q = rewards + self.config["gamma"] * next_q_values * (1 - dones)
                    
                    current_q = self.critic(torch.cat([states, actions], dim=1))
                    critic_loss = self.criterion(current_q, target_q)
                    
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
                    
                    means, stds = self.actor(states)
                    sampled_actions = torch.clamp(torch.distributions.Normal(means, stds).rsample(), -1, 1)
                    actor_loss = -self.critic(torch.cat([states, sampled_actions], dim=1)).mean()
                    
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                    
                    self.soft_update_target_networks()
                    
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
            
            avg_rewards.append(episode_reward)
            
            pbar.set_postfix({
                "Running Average Reward": f"{sum(avg_rewards)/len(avg_rewards)}",
                "Actor Loss": f"{actor_losses[-1] if actor_losses else 0:.4f}",
                "Critic Loss": f"{critic_losses[-1] if critic_losses else 0:.4f}"
            })
        
        if path:
            self.actor.save(os.path.join(path, "c_lander.pth"))
            self.critic.save(os.path.join(path, "c_critic.pth"))
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(avg_rewards, label="Reward")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Reward per Episode")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(actor_losses, label="Actor Loss", alpha=0.7)
            plt.plot(critic_losses, label="Critic Loss", alpha=0.7)
            plt.xlabel("Update Step")
            plt.ylabel("Loss")
            plt.title("Actor & Critic Loss")
            plt.legend()

            plt.tight_layout()
            plot_path = os.path.join(path, "c_training_curve.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Training curve saved to {plot_path}")

    def test_continuous(self, path: str):
        self.actor.load(os.path.join(path, "c_lander.pth"))
        
        video_path = os.path.join(path, "c_lander_video.mp4")
        
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
                mean, _ = self.actor(torch.tensor(state).float().unsqueeze(0).to(self.device))
                action = mean.squeeze(0).cpu().numpy()
            
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