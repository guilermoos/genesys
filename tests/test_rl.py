"""
Tests for RL components.
"""

import pytest
import numpy as np
import torch

from app.rl.network import DQNNetwork, DuelingDQNNetwork
from app.rl.replay_buffer import ReplayBuffer, Experience
from app.rl.agent import DQNAgent


class TestDQNNetwork:
    """Tests for DQN network."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = DQNNetwork(state_size=10, action_size=4)
        
        assert network.state_size == 10
        assert network.action_size == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        network = DQNNetwork(state_size=10, action_size=4)
        state = torch.randn(1, 10)
        
        q_values = network(state)
        
        assert q_values.shape == (1, 4)
    
    def test_batch_forward(self):
        """Test forward pass with batch."""
        network = DQNNetwork(state_size=10, action_size=4)
        states = torch.randn(32, 10)
        
        q_values = network(states)
        
        assert q_values.shape == (32, 4)


class TestReplayBuffer:
    """Tests for replay buffer."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(capacity=1000)
        
        assert len(buffer) == 0
    
    def test_add_experience(self):
        """Test adding experience."""
        buffer = ReplayBuffer(capacity=1000)
        
        state = np.zeros(10)
        action = 0
        reward = 1.0
        next_state = np.zeros(10)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
    
    def test_sample(self):
        """Test sampling experiences."""
        buffer = ReplayBuffer(capacity=1000)
        
        # Add multiple experiences
        for i in range(100):
            buffer.add(
                np.zeros(10),
                i % 4,
                float(i),
                np.zeros(10),
                i % 10 == 0,
            )
        
        states, actions, rewards, next_states, dones = buffer.sample(32)
        
        assert states.shape == (32, 10)
        assert actions.shape == (32, 1)
        assert rewards.shape == (32, 1)
        assert next_states.shape == (32, 10)
        assert dones.shape == (32, 1)
    
    def test_sample_not_ready(self):
        """Test sampling when not enough experiences."""
        buffer = ReplayBuffer(capacity=1000)
        buffer.add(np.zeros(10), 0, 1.0, np.zeros(10), False)
        
        with pytest.raises(ValueError):
            buffer.sample(32)


class TestDQNAgent:
    """Tests for DQN agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(state_size=10, action_size=4)
        
        assert agent.state_size == 10
        assert agent.action_size == 4
        assert agent.epsilon == 1.0
    
    def test_get_action_training(self):
        """Test action selection in training mode."""
        agent = DQNAgent(state_size=10, action_size=4)
        state = np.zeros(10)
        
        action = agent.get_action(state, training=True)
        
        assert 0 <= action < 4
    
    def test_get_action_eval(self):
        """Test action selection in eval mode."""
        agent = DQNAgent(state_size=10, action_size=4)
        state = np.zeros(10)
        
        action = agent.get_action(state, training=False)
        
        assert 0 <= action < 4
    
    def test_store_experience(self):
        """Test storing experience."""
        agent = DQNAgent(state_size=10, action_size=4)
        
        agent.store_experience(
            np.zeros(10),
            0,
            1.0,
            np.zeros(10),
            False,
        )
        
        assert len(agent.replay_buffer) == 1
    
    def test_learn_not_ready(self):
        """Test learning when buffer not ready."""
        agent = DQNAgent(state_size=10, action_size=4, batch_size=32)
        
        loss = agent.learn()
        
        assert loss is None
    
    def test_save_load(self, tmp_path):
        """Test saving and loading agent."""
        agent = DQNAgent(state_size=10, action_size=4)
        
        # Add some experiences and learn
        for i in range(100):
            agent.store_experience(
                np.zeros(10),
                i % 4,
                float(i),
                np.zeros(10),
                False,
            )
        agent.learn()
        
        # Save
        save_path = tmp_path / "agent.pt"
        agent.save(str(save_path))
        
        # Load
        new_agent = DQNAgent(state_size=10, action_size=4)
        new_agent.load(str(save_path))
        
        assert new_agent.train_step == agent.train_step
