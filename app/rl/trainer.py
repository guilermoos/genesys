"""
Training orchestrator for DQN agents.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import time
import os

import numpy as np
import torch

from app.rl.agent import DQNAgent
from app.templates.base import BaseTemplate


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    
    def add_episode(self, reward: float, length: int, epsilon: float) -> None:
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilon_history.append(epsilon)
    
    def add_loss(self, loss: float) -> None:
        """Add training loss."""
        if loss is not None:
            self.losses.append(loss)
    
    @property
    def avg_reward_last_100(self) -> float:
        """Average reward over last 100 episodes."""
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards[-100:])
    
    @property
    def best_reward(self) -> float:
        """Best reward achieved."""
        if not self.episode_rewards:
            return 0.0
        return max(self.episode_rewards)
    
    @property
    def avg_loss_last_100(self) -> float:
        """Average loss over last 100 updates."""
        if not self.losses:
            return 0.0
        return np.mean(self.losses[-100:])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
            "epsilon_history": self.epsilon_history,
            "avg_reward_last_100": self.avg_reward_last_100,
            "best_reward": self.best_reward,
            "avg_loss_last_100": self.avg_loss_last_100,
            "total_episodes": len(self.episode_rewards),
        }


class Trainer:
    """
    Training orchestrator for DQN agents.
    
    Manages the training loop, metrics collection, and checkpointing.
    """
    
    def __init__(
        self,
        agent: DQNAgent,
        environment: BaseTemplate,
        save_dir: str,
        checkpoint_freq: int = 100,
        log_freq: int = 10,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            agent: DQN agent to train
            environment: Environment template
            save_dir: Directory to save checkpoints
            checkpoint_freq: Episodes between checkpoints
            log_freq: Episodes between logging
            progress_callback: Optional callback for progress updates
        """
        self.agent = agent
        self.environment = environment
        self.save_dir = save_dir
        self.checkpoint_freq = checkpoint_freq
        self.log_freq = log_freq
        self.progress_callback = progress_callback
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Metrics
        self.metrics = TrainingMetrics()
        self.start_time: Optional[float] = None
        self.is_training: bool = False
        self.current_episode: int = 0
    
    def train(
        self,
        num_episodes: int,
        max_steps_per_episode: int = None,
    ) -> TrainingMetrics:
        """
        Train the agent for specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            TrainingMetrics with all collected metrics
        """
        if max_steps_per_episode is None:
            max_steps_per_episode = self.environment.config.get("max_steps", 500)
        
        self.is_training = True
        self.start_time = time.time()
        
        try:
            for episode in range(num_episodes):
                if not self.is_training:
                    break
                
                self.current_episode = episode
                
                # Run one episode
                episode_reward, episode_length = self._run_episode(max_steps_per_episode)
                
                # Record metrics
                self.metrics.add_episode(
                    episode_reward,
                    episode_length,
                    self.agent.epsilon,
                )
                
                # Log progress
                if (episode + 1) % self.log_freq == 0:
                    self._log_progress(episode + 1, num_episodes)
                
                # Save checkpoint
                if (episode + 1) % self.checkpoint_freq == 0:
                    self._save_checkpoint(episode + 1)
                
                # Call progress callback
                if self.progress_callback:
                    self.progress_callback({
                        "episode": episode + 1,
                        "total_episodes": num_episodes,
                        "reward": episode_reward,
                        "avg_reward_100": self.metrics.avg_reward_last_100,
                        "epsilon": self.agent.epsilon,
                    })
        
        except Exception as e:
            print(f"Training interrupted: {e}")
            raise
        
        finally:
            self.is_training = False
        
        return self.metrics
    
    def _run_episode(self, max_steps: int) -> tuple:
        """
        Run a single training episode.
        
        Args:
            max_steps: Maximum steps in episode
            
        Returns:
            Tuple of (total_reward, episode_length)
        """
        state = self.environment.reset()
        total_reward = 0.0
        
        for step in range(max_steps):
            # Select action
            action = self.agent.get_action(state, training=True)
            
            # Execute action
            next_state, reward, done, info = self.environment.step(action)
            
            # Store experience
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # Learn from experiences
            loss = self.agent.learn()
            self.metrics.add_loss(loss)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        return total_reward, step + 1
    
    def _log_progress(self, current_episode: int, total_episodes: int) -> None:
        """Log training progress."""
        elapsed = time.time() - self.start_time
        episodes_per_sec = current_episode / elapsed if elapsed > 0 else 0
        
        print(
            f"Episode {current_episode}/{total_episodes} | "
            f"Avg Reward (100): {self.metrics.avg_reward_last_100:.2f} | "
            f"Best: {self.metrics.best_reward:.2f} | "
            f"Epsilon: {self.agent.epsilon:.3f} | "
            f"Loss: {self.metrics.avg_loss_last_100:.4f} | "
            f"Speed: {episodes_per_sec:.1f} eps/s"
        )
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.save_dir,
            f"checkpoint_episode_{episode}.pt"
        )
        self.agent.save(checkpoint_path)
    
    def save_final_model(self, filename: str = "final_model.pt") -> str:
        """
        Save final trained model.
        
        Args:
            filename: Name of the model file
            
        Returns:
            Path to saved model
        """
        model_path = os.path.join(self.save_dir, filename)
        self.agent.save(model_path)
        return model_path
    
    def stop(self) -> None:
        """Stop training gracefully."""
        self.is_training = False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training session."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            "total_episodes": len(self.metrics.episode_rewards),
            "total_steps": sum(self.metrics.episode_lengths),
            "avg_reward": np.mean(self.metrics.episode_rewards) if self.metrics.episode_rewards else 0,
            "best_reward": self.metrics.best_reward,
            "final_epsilon": self.agent.epsilon,
            "training_duration_seconds": elapsed,
            "episodes_per_second": len(self.metrics.episode_rewards) / elapsed if elapsed > 0 else 0,
        }
