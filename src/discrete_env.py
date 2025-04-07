import cv2 
import gym
from model import DiscreteQN
from replay import ReplayBuffer
import random 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import yaml 
import os 
from tqdm import tqdm

class D_environment: 
    def __init__(self, config_path: str, weights: str = None): 
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
        self.env = gym.make("LunarLander-v2", render_mode="rgb_array")
        self.model = DiscreteQN(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.model_target = DiscreteQN(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(self.config['max_memory'])
        
        self.target_update_freq = self.config['target_update_steps']
        
        self.epsilon = self.config['epsilon']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']

        if weights:
            self.model.load(weights)
            self.model_target.load(weights)
            
    def update_target(self):
        self.model_target.load_state_dict(self.model.state_dict())
        
    def train_dqn(self, path: str = None): 
        avg_awards = []
        avg_grad = [0]
        avg_loss = [0]
        
        pbar = tqdm(range(self.config["episode"]))
        steps = 0
        for episode in pbar: 
            state, _ = self.env.reset()
            total_reward = 0
            done = False 
            
            while not done: 
                if random.random() < self.epsilon: 
                    action = self.env.action_space.sample()
                else: 
                    q_values = self.model(torch.tensor(state).float().to(self.device))
                    action = torch.argmax(q_values).item()
                    
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward 
                
                if len(self.replay_buffer) > self.config["batch_size"]: 
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config["batch_size"])
                    with torch.no_grad(): 
                        next_q_values = self.model_target(next_states.to(self.device)).max(dim=1)[0].unsqueeze(1).detach().cpu()
                        targets = rewards + self.config["gamma"] * next_q_values * (1 - dones)
                        targets = targets.to(self.device)
                        
                    current_q_values = self.model(states.to(self.device)).gather(1, actions.to(self.device))
                    loss = self.criterion(targets, current_q_values)
                    self.optimizer.zero_grad() 
                    loss.backward() 

                    total_grad = 0.0 
                    for name, param in self.model.named_parameters(): 
                        if param.grad is not None: 
                            total_grad += param.grad.norm().item()
                            
                    avg_grad.append(total_grad)
                    
                    self.optimizer.step()
                    
                    avg_loss.append(loss.item())

                steps += 1
                    
                if steps % self.target_update_freq == 0: 
                    self.update_target()
                
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            avg_awards.append(total_reward)
            
            pbar.postfix = f"Avg Reward: {sum(avg_awards) / len(avg_awards):.4f}, Epsilon: {self.epsilon:.4f}, Loss: {sum(avg_loss)/len(avg_loss):.4f}, Grad: {sum(avg_grad)/len(avg_grad):.4f}"
            
        self.model.save(os.path.join(path, "d_lander.pth"))
    
    def test_dqn(self, path: str):
        self.model.load(os.path.join(path, 'd_lander.pth'))
        
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
            
            q_values = self.model(torch.tensor(state).float().to(self.device))
            action = torch.argmax(q_values).item()
            
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