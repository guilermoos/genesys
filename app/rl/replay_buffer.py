"""
Replay buffer for experience replay in DQN.
"""

from typing import Tuple, NamedTuple
from collections import deque
import random

import numpy as np
import torch


class Experience(NamedTuple):
    """Single experience tuple."""
    
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    
    Implements experience replay for DQN training, which helps break
    correlations between consecutive samples and improves sample efficiency.
    """
    
    def __init__(self, capacity: int, seed: int = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Randomly sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough experiences in buffer. "
                f"Have {len(self.buffer)}, requested {batch_size}"
            )
        
        experiences = random.sample(self.buffer, batch_size)
        
        # Unpack experiences
        states = np.vstack([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences]).astype(np.float32)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Samples experiences with probability proportional to their TD error,
    focusing learning on more "surprising" experiences.
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        seed: int = None,
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction
            beta_frames: Number of frames to anneal beta to 1
            seed: Random seed
        """
        super().__init__(capacity, seed)
        
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Priorities stored separately
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    @property
    def beta(self) -> float:
        """Calculate current beta value for importance sampling."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience with maximum priority."""
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        
        super().add(state, action, reward, next_state, done)
        
        # Set priority for new experience
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample experiences based on priorities.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).unsqueeze(1)
        
        self.frame += 1
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        states = np.vstack([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences]).astype(np.float32)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant for stability
