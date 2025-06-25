"""
td3_agent.py

This module implements the TD3 (Twin Delayed DDPG) reinforcement learning agent
for safe navigation. The agent learns to reach goals while its actions are
filtered through the SafeLLMRA reachability analysis for safety.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


class Actor(nn.Module):
    """Actor network for TD3 agent."""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: np.ndarray, device: torch.device):
        super(Actor, self).__init__()
        self.device = device
        self.max_action = torch.FloatTensor(max_action).to(device)  # Move to device
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a * self.max_action


class Critic(nn.Module):
    """Critic network for TD3 agent."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1


class ReplayBuffer:
    """Experience replay buffer for TD3."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class TD3Agent:
    """
    TD3 (Twin Delayed DDPG) agent for safe reinforcement learning.
    The agent learns to navigate to goals while being safety-filtered.
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: np.ndarray, 
                 min_action: np.ndarray, config: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TD3 Agent using device: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        
        # Hyperparameters
        self.lr = config.get("training", {}).get("learning_rate", 3e-4)
        self.batch_size = config.get("training", {}).get("batch_size", 256)
        self.gamma = config.get("training", {}).get("gamma", 0.99)
        self.tau = config.get("training", {}).get("tau", 0.005)
        self.policy_noise = config.get("training", {}).get("policy_noise", 0.2)
        self.noise_clip = config.get("training", {}).get("noise_clip", 0.5)
        self.policy_freq = config.get("training", {}).get("policy_freq", 2)
        
        # Networks - pass device to Actor
        self.actor = Actor(state_dim, action_dim, max_action, self.device).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, self.device).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Replay buffer
        buffer_size = config.get("training", {}).get("replay_buffer_size", 100000)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training variables
        self.total_it = 0
        self.exploration_noise = 0.1
        
        # Goal and navigation parameters
        self.goal_position = None
        self.goal_tolerance = 0.2  # Distance tolerance to consider goal reached
        
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using the actor network."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
        
        # Clip to action bounds
        action = np.clip(action, self.min_action, self.max_action)
        return action
    
    def set_goal(self, goal_position: np.ndarray):
        """Set a new goal position for the agent."""
        self.goal_position = goal_position.copy()
        logger.info(f"New goal set: {self.goal_position}")
    
    def compute_reward(self, current_state: np.ndarray, action: np.ndarray, 
                      next_state: np.ndarray, goal_reached: bool, 
                      collision_occurred: bool) -> float:
        """Compute reward based on goal progress and safety."""
        if self.goal_position is None:
            return 0.0
        
        current_pos = current_state[:2]  # x, y position
        next_pos = next_state[:2]
        goal_pos = self.goal_position[:2]
        
        # Distance rewards
        current_distance = np.linalg.norm(current_pos - goal_pos)
        next_distance = np.linalg.norm(next_pos - goal_pos)
        progress_reward = current_distance - next_distance  # Positive if moving towards goal
        
        # Goal reached reward
        goal_reward = 100.0 if goal_reached else 0.0
        
        # Safety penalty
        collision_penalty = -50.0 if collision_occurred else 0.0
        
        # Action regularization (encourage smooth actions)
        action_penalty = -0.01 * np.sum(np.square(action))
        
        # Distance penalty (encourage staying close to goal)
        distance_penalty = -0.1 * next_distance
        
        total_reward = (
            10.0 * progress_reward +
            goal_reward +
            collision_penalty +
            action_penalty +
            distance_penalty
        )
        
        return total_reward
    
    def is_goal_reached(self, current_state: np.ndarray) -> bool:
        """Check if the current state is close enough to the goal."""
        if self.goal_position is None:
            return False
        
        current_pos = current_state[:2]
        goal_pos = self.goal_position[:2]
        distance = np.linalg.norm(current_pos - goal_pos)
        
        return distance < self.goal_tolerance
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> dict:
        """Train the TD3 agent."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        self.total_it += 1
        
        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                torch.FloatTensor(self.min_action).to(self.device),
                torch.FloatTensor(self.max_action).to(self.device))
            
            # Compute target Q values
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        training_info = {
            "critic_loss": critic_loss.item(),
            "q1_mean": current_Q1.mean().item(),
            "q2_mean": current_Q2.mean().item()
        }
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            training_info["actor_loss"] = actor_loss.item()
        
        return training_info
    
    def save(self, filepath: str):
        """Save the trained models."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        logger.info(f"TD3 model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained models."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Update target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        logger.info(f"TD3 model loaded from {filepath}")
