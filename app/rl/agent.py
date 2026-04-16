"""
DQN Agent implementation with epsilon-greedy policy and target network.
"""

from typing import List, Tuple, Optional
import random
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from app.rl.network import DQNNetwork
from app.rl.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent.
    
    Implements DQN algorithm with:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = None,
        seed: int = None,
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_layers: Sizes of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to use ('cpu', 'cuda', or None for auto)
            seed: Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers or [128, 128]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set random seeds
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Q-Networks
        self.q_network = DQNNetwork(
            state_size, action_size, self.hidden_layers
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            state_size, action_size, self.hidden_layers
        ).to(self.device)
        
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, seed)
        
        # Training metrics
        self.train_step = 0
        self.loss_history = []
    
    def update_target_network(self) -> None:
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon)
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randrange(self.action_size)
        
        # Greedy action (exploitation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax(dim=1).item()
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in given state.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for each action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy()[0]
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def learn(self) -> Optional[float]:
        """
        Perform one step of learning from replay buffer.
        
        Returns:
            Loss value if learning occurred, None otherwise
        """
        # Check if enough experiences
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q(s_t, a) - current Q values
        current_q = self.q_network.get_action_q_values(states, actions)
        
        # Compute Q(s_{t+1}, a) for all next states
        with torch.no_grad():
            # Double DQN: use online network to select action
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # Use target network to evaluate action
            next_q = self.target_network.get_action_q_values(next_states, next_actions)
            
            # Compute target Q value
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        # Record loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def save(self, path: str) -> None:
        """
        Save agent state to file.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_step": self.train_step,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """
        Load agent state from file.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
        self.train_step = checkpoint["train_step"]
    
    def get_config(self) -> dict:
        """Get agent configuration."""
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "buffer_size": self.replay_buffer.capacity,
        }
