"""
Decision Optimization template - A generic decision-making environment.

Useful for problems where an agent must choose discrete actions to maximize
a configurable objective function.
"""

from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np

from app.templates.base import BaseTemplate


class DecisionOptimizationTemplate(BaseTemplate):
    """
    Decision optimization environment template.
    
    This template provides a flexible environment for decision-making problems
    with configurable state dimensions, action space, and reward functions.
    
    State: Configurable vector of features
    Actions: Discrete actions from configurable set
    Reward: Based on configurable objective function
    """
    
    name = "decision_optimization"
    description = "Generic decision optimization with configurable reward"
    version = "1.0.0"
    
    # Built-in reward functions
    REWARD_FUNCTIONS = {
        "linear": "_linear_reward",
        "quadratic": "_quadratic_reward",
        "sparse": "_sparse_reward",
        "custom": "_custom_reward",
    }
    
    def __init__(self, config: Dict[str, Any]):
        # Configuration defaults
        self.state_size: int = 10
        self.action_space: List[int] = [0, 1, 2]
        self.max_steps: int = 100
        self.reward_type: str = "linear"
        self.reward_params: Dict[str, Any] = {}
        self.state_change_prob: float = 0.1
        self.noise_std: float = 0.0
        
        # Runtime state
        self.state: np.ndarray = np.zeros(10)
        self.steps: int = 0
        self.episode: int = 0
        self.best_action_history: List[int] = []
        
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """Validate decision optimization configuration."""
        # Validate state_size
        if "state_size" in self.config:
            size = self.config["state_size"]
            if not isinstance(size, int) or size < 1 or size > 1000:
                raise ValueError("state_size must be an integer between 1 and 1000")
            self.state_size = size
        
        # Validate action_space
        if "action_space" in self.config:
            actions = self.config["action_space"]
            if not isinstance(actions, list) or len(actions) < 1:
                raise ValueError("action_space must be a non-empty list")
            if not all(isinstance(a, int) and a >= 0 for a in actions):
                raise ValueError("action_space must contain non-negative integers")
            if len(set(actions)) != len(actions):
                raise ValueError("action_space must contain unique values")
            self.action_space = sorted(actions)
        
        # Validate max_steps
        if "max_steps" in self.config:
            steps = self.config["max_steps"]
            if not isinstance(steps, int) or steps < 1:
                raise ValueError("max_steps must be a positive integer")
            self.max_steps = steps
        
        # Validate reward_type
        if "reward_type" in self.config:
            reward_type = self.config["reward_type"]
            if reward_type not in self.REWARD_FUNCTIONS:
                valid_types = ", ".join(self.REWARD_FUNCTIONS.keys())
                raise ValueError(f"reward_type must be one of: {valid_types}")
            self.reward_type = reward_type
        
        # Validate reward_params
        if "reward_params" in self.config:
            self.reward_params = self.config["reward_params"]
        
        # Validate state_change_prob
        if "state_change_prob" in self.config:
            prob = self.config["state_change_prob"]
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
                raise ValueError("state_change_prob must be between 0 and 1")
            self.state_change_prob = prob
        
        # Validate noise_std
        if "noise_std" in self.config:
            noise = self.config["noise_std"]
            if not isinstance(noise, (int, float)) or noise < 0:
                raise ValueError("noise_std must be non-negative")
            self.noise_std = noise
    
    def _setup_environment(self) -> None:
        """Setup the decision optimization environment."""
        self.state = self._generate_initial_state()
        self.steps = 0
        self.episode = 0
    
    def _generate_initial_state(self) -> np.ndarray:
        """Generate initial state vector."""
        # Random state with some structure
        state = np.random.randn(self.state_size).astype(np.float32)
        # Normalize to reasonable range
        state = np.clip(state, -3, 3) / 3
        return state
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = self._generate_initial_state()
        self.steps = 0
        self.episode += 1
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take from action_space
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if action not in self.action_space:
            raise ValueError(
                f"Invalid action {action}. Valid actions: {self.action_space}"
            )
        
        self.steps += 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Add noise to reward if configured
        if self.noise_std > 0:
            reward += np.random.normal(0, self.noise_std)
        
        # State transition
        if np.random.random() < self.state_change_prob:
            # Significant state change
            self.state = self._generate_initial_state()
        else:
            # Small gradual change
            self.state += np.random.randn(self.state_size).astype(np.float32) * 0.1
            self.state = np.clip(self.state, -1, 1)
        
        # Check if episode is done
        done = self.steps >= self.max_steps
        
        # Track best action
        best_action = self._get_best_action()
        is_optimal = action == best_action
        
        info = {
            "steps": self.steps,
            "episode": self.episode,
            "is_optimal_action": is_optimal,
            "best_action": best_action,
            "state_mean": float(np.mean(self.state)),
            "state_std": float(np.std(self.state)),
        }
        
        return self.state.copy(), float(reward), done, info
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for the given action."""
        reward_method = getattr(self, self.REWARD_FUNCTIONS[self.reward_type])
        return reward_method(action)
    
    def _linear_reward(self, action: int) -> float:
        """
        Linear reward function.
        
        Reward is proportional to action value and state features.
        Higher actions generally yield higher rewards.
        """
        base_reward = action * 0.5
        state_contribution = np.mean(self.state) * 2
        return base_reward + state_contribution
    
    def _quadratic_reward(self, action: int) -> float:
        """
        Quadratic reward function.
        
        Reward peaks at optimal action based on current state.
        """
        # Optimal action depends on state
        optimal = int(np.clip(
            (np.mean(self.state) + 1) / 2 * (len(self.action_space) - 1),
            0,
            len(self.action_space) - 1
        ))
        optimal_action = self.action_space[optimal]
        
        # Quadratic penalty for distance from optimal
        distance = abs(action - optimal_action)
        max_distance = max(self.action_space) - min(self.action_space)
        normalized_distance = distance / max_distance if max_distance > 0 else 0
        
        return 10 * (1 - normalized_distance ** 2)
    
    def _sparse_reward(self, action: int) -> float:
        """
        Sparse reward function.
        
        Only gives positive reward for optimal action.
        """
        best_action = self._get_best_action()
        return 1.0 if action == best_action else 0.0
    
    def _custom_reward(self, action: int) -> float:
        """
        Custom reward function based on reward_params.
        
        Supports weighted sum of state features with action-dependent weights.
        """
        weights = self.reward_params.get("weights", {})
        action_weights = weights.get(str(action), [1.0] * self.state_size)
        
        if len(action_weights) != self.state_size:
            action_weights = [1.0] * self.state_size
        
        reward = float(np.dot(self.state, action_weights))
        
        # Add bias if specified
        bias = self.reward_params.get("bias", 0)
        reward += bias
        
        return reward
    
    def _get_best_action(self) -> int:
        """Determine the best action for current state."""
        best_action = self.action_space[0]
        best_reward = float('-inf')
        
        for action in self.action_space:
            reward = self._calculate_reward(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        return best_action
    
    def get_state_size(self) -> int:
        """Get the size of the state vector."""
        return self.state_size
    
    def get_action_space(self) -> List[int]:
        """Get list of valid actions."""
        return self.action_space
    
    def get_action_size(self) -> int:
        """Get number of possible actions."""
        return len(self.action_space)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this template."""
        return {
            "state_size": 10,
            "action_space": [0, 1, 2],
            "max_steps": 100,
            "reward_type": "linear",
            "reward_params": {},
            "state_change_prob": 0.1,
            "noise_std": 0.0,
        }
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get JSON schema for template configuration."""
        return {
            "type": "object",
            "properties": {
                "state_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "Dimensionality of state vector",
                },
                "action_space": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 1,
                    "description": "List of valid actions",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum steps per episode",
                },
                "reward_type": {
                    "type": "string",
                    "enum": list(self.REWARD_FUNCTIONS.keys()),
                    "description": "Type of reward function",
                },
                "reward_params": {
                    "type": "object",
                    "description": "Parameters for custom reward function",
                },
                "state_change_prob": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Probability of state reset after each step",
                },
                "noise_std": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Standard deviation of reward noise",
                },
            },
            "required": [],
        }
