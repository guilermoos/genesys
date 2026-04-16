"""
Deep Q-Network (DQN) neural network implementation.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for approximating Q-values.
    
    Architecture:
    - Input: state vector
    - Hidden layers: configurable fully connected layers
    - Output: Q-value for each action
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = None,
        dropout: float = 0.0,
    ):
        """
        Initialize DQN network.
        
        Args:
            state_size: Dimension of input state
            action_size: Number of possible actions
            hidden_layers: List of hidden layer sizes (default: [128, 128])
            dropout: Dropout probability for regularization
        """
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        if hidden_layers is None:
            hidden_layers = [128, 128]
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_size)
            
        Returns:
            Q-values tensor of shape (batch_size, action_size)
        """
        return self.network(state)
    
    def get_action_q_values(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Get Q-value for a specific action.
        
        Args:
            state: State tensor of shape (batch_size, state_size)
            action: Action tensor of shape (batch_size, 1)
            
        Returns:
            Q-values for the specified actions
        """
        q_values = self.forward(state)
        return q_values.gather(1, action)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    
    This can lead to better performance in some environments by learning
    which states are valuable independently of actions.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = None,
        value_stream_size: int = 64,
        advantage_stream_size: int = 64,
    ):
        """
        Initialize Dueling DQN network.
        
        Args:
            state_size: Dimension of input state
            action_size: Number of possible actions
            hidden_layers: List of shared hidden layer sizes
            value_stream_size: Size of value stream hidden layer
            advantage_stream_size: Size of advantage stream hidden layer
        """
        super(DuelingDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        if hidden_layers is None:
            hidden_layers = [128, 128]
        
        # Shared feature layers
        feature_layers = []
        prev_size = state_size
        
        for hidden_size in hidden_layers:
            feature_layers.append(nn.Linear(prev_size, hidden_size))
            feature_layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.feature = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, value_stream_size),
            nn.ReLU(),
            nn.Linear(value_stream_size, 1),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, advantage_stream_size),
            nn.ReLU(),
            nn.Linear(advantage_stream_size, action_size),
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling network.
        
        Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        
        Args:
            state: State tensor
            
        Returns:
            Q-values tensor
        """
        features = self.feature(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
