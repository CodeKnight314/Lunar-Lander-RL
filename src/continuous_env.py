import cv2 
import gym
import torch.distributions
from model import ActorContinuous, Critic

import torch 
import torch.nn as nn 
import torch.optim as optim
import yaml
import os
from tqdm import tqdm

class C_environment:
    def __init__(self, config_path: str, actor_weights: str = None, critic_weights: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        
        self.env = gym.make("LunarLanderContinuous-v2", render_mode="rgb_array")
        self.actor = ActorContinuous(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(self.device)
        self.critic = Critic(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(self.device)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.config["critic_lr"])
        
        self.criterion = nn.MSELoss()
        
        self.n_step = self.config["n_step"]
        
        if actor_weights: 
            self.actor.load(actor_weights)
        if critic_weights: 
            self.critic.load(critic_weights)
            
            
    def train_continuous(self, path: str = None):
        avg_awards = []
        avg_loss = [0]
        
        pbar = tqdm(range(self.config["episode"]), desc="[Episode]: ")
        for epsiode in pbar: 
            state, _ = self.env.reset()
            done = False
            
            while not done:  
                trajectory = {"states": [], 
                              "actions": [], 
                              "rewards": [],
                              "dones": [], 
                              "log_probs": [], 
                              "values": []}
                
                for _ in range(self.n_step): 
                    state_tensor = torch.tensor(state).float().to(self.device)
                    mean, std = self.actor(state_tensor)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    value = self.critic(state_tensor)
                    
                    next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                    done = terminated or truncated
                    
                    trajectory["states"].append(state_tensor)
                    trajectory["actions"].append(action)
                    trajectory["rewards"].append(torch.tensor(reward, dtype=torch.float32))
                    trajectory["dones"].append(torch.tensor(done, dtype=torch.float32))
                    trajectory["log_probs"].append(log_prob)
                    trajectory["values"].append(value)
                    
                    state = next_state
                    if done: 
                        break
                    
                if done: 
                    next_value = torch.tensor(0.0).to(self.device)
                else:
                    next_state_tensor = torch.tensor(next_state).float().to(self.device)
                    next_value = self.critic(next_state_tensor).detach()
                    
                returns = [] 
                R = next_value 
                for reward, done_flag in zip(reversed(trajectory["rewards"]), reversed(trajectory["dones"])):
                    R = reward + self.config["gamma"] * R * (1 - done_flag)
                    returns.insert(0, R)
                    
                returns = torch.stack(returns)
                values = torch.stack(trajectory["values"])
                log_probs = torch.stack(trajectory["log_probs"])
                
                advantages = returns - values.detach()

                critic_loss = ((returns - values) ** 2).mean()
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                
                actor_loss = -(log_probs * advantages).mean() 
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                avg_loss.append(actor_loss.item())
                avg_loss.append(critic_loss.item())
                avg_awards.append(sum(trajectory["rewards"]).item())

            pbar.postfix = f"Avg Reward: {sum(avg_awards) / len(avg_awards):.4f}, Loss: {sum(avg_loss)/len(avg_loss):.4f}"
        
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
                mean, std = self.actor(torch.tensor(state).float().to(self.device))
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
                    