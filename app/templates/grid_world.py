"""
GridWorld template - A simple grid navigation environment.

The agent must navigate from start to goal in a grid, avoiding obstacles.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from app.templates.base import BaseTemplate


class GridWorldTemplate(BaseTemplate):
    """
    GridWorld environment template.
    
    State: [agent_x, agent_y, goal_x, goal_y] + obstacle proximity sensors
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    Reward: -1 per step, +10 for reaching goal, -5 for hitting obstacle
    """
    
    name = "grid_world"
    description = "Grid navigation environment with obstacles"
    version = "1.0.0"
    
    # Action mappings
    ACTIONS = {
        0: "UP",
        1: "RIGHT",
        2: "DOWN",
        3: "LEFT",
    }
    
    DIRECTIONS = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1),  # LEFT
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.grid_size: Tuple[int, int] = (5, 5)
        self.obstacle_count: int = 3
        self.max_steps: int = 100
        self.use_sensors: bool = True
        self.sensor_range: int = 2
        
        # Runtime state
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (4, 4)
        self.obstacles: List[Tuple[int, int]] = []
        self.steps: int = 0
        
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """Validate grid world configuration."""
        if "grid_size" in self.config:
            size = self.config["grid_size"]
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                raise ValueError("grid_size must be a list/tuple of 2 integers")
            if size[0] < 3 or size[1] < 3:
                raise ValueError("grid_size must be at least 3x3")
            self.grid_size = tuple(size)
        
        if "obstacle_count" in self.config:
            count = self.config["obstacle_count"]
            max_obstacles = self.grid_size[0] * self.grid_size[1] - 2
            if not isinstance(count, int) or count < 0 or count > max_obstacles:
                raise ValueError(f"obstacle_count must be between 0 and {max_obstacles}")
            self.obstacle_count = count
        
        if "max_steps" in self.config:
            steps = self.config["max_steps"]
            if not isinstance(steps, int) or steps < 10:
                raise ValueError("max_steps must be at least 10")
            self.max_steps = steps
        
        if "use_sensors" in self.config:
            self.use_sensors = bool(self.config["use_sensors"])
        
        if "sensor_range" in self.config:
            range_val = self.config["sensor_range"]
            if not isinstance(range_val, int) or range_val < 1:
                raise ValueError("sensor_range must be a positive integer")
            self.sensor_range = range_val
    
    def _setup_environment(self) -> None:
        """Setup the grid world environment."""
        self._generate_obstacles()
    
    def _generate_obstacles(self) -> None:
        """Generate random obstacles avoiding start and goal positions."""
        self.obstacles = []
        all_positions = [
            (r, c) 
            for r in range(self.grid_size[0]) 
            for c in range(self.grid_size[1])
        ]
        
        # Exclude start and goal positions
        excluded = {(0, 0), (self.grid_size[0] - 1, self.grid_size[1] - 1)}
        available = [p for p in all_positions if p not in excluded]
        
        # Randomly select obstacle positions
        if self.obstacle_count > 0 and available:
            indices = np.random.choice(
                len(available), 
                size=min(self.obstacle_count, len(available)), 
                replace=False
            )
            self.obstacles = [available[i] for i in indices]
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size[0] - 1, self.grid_size[1] - 1)
        self.steps = 0
        self._generate_obstacles()
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Valid actions: {list(self.ACTIONS.keys())}")
        
        self.steps += 1
        
        # Calculate new position
        dr, dc = self.DIRECTIONS[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        new_pos = (new_row, new_col)
        
        # Check bounds
        if (new_row < 0 or new_row >= self.grid_size[0] or 
            new_col < 0 or new_col >= self.grid_size[1]):
            # Hit wall
            reward = -1.0
            done = False
        elif new_pos in self.obstacles:
            # Hit obstacle
            reward = -5.0
            done = False
        else:
            # Valid move
            self.agent_pos = new_pos
            
            # Check if reached goal
            if self.agent_pos == self.goal_pos:
                reward = 10.0
                done = True
            else:
                # Small penalty for each step
                reward = -0.1
                done = False
        
        # Check max steps
        if self.steps >= self.max_steps:
            done = True
        
        info = {
            "steps": self.steps,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "action_taken": self.ACTIONS[action],
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        State includes:
        - Normalized agent position (2 values)
        - Normalized goal position (2 values)
        - Sensor readings for obstacles (4-8 values depending on sensor range)
        """
        # Base state: normalized positions
        state = [
            self.agent_pos[0] / self.grid_size[0],
            self.agent_pos[1] / self.grid_size[1],
            self.goal_pos[0] / self.grid_size[0],
            self.goal_pos[1] / self.grid_size[1],
        ]
        
        # Add sensor readings if enabled
        if self.use_sensors:
            sensors = self._get_sensor_readings()
            state.extend(sensors)
        
        return np.array(state, dtype=np.float32)
    
    def _get_sensor_readings(self) -> List[float]:
        """
        Get obstacle sensor readings in 4 directions.
        
        Returns normalized distances to nearest obstacle in each direction.
        """
        sensors = []
        
        for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # UP, RIGHT, DOWN, LEFT
            distance = self.sensor_range
            
            for d in range(1, self.sensor_range + 1):
                check_pos = (
                    self.agent_pos[0] + direction[0] * d,
                    self.agent_pos[1] + direction[1] * d,
                )
                
                # Check bounds or obstacle
                if (check_pos[0] < 0 or check_pos[0] >= self.grid_size[0] or
                    check_pos[1] < 0 or check_pos[1] >= self.grid_size[1]):
                    distance = d - 1
                    break
                
                if check_pos in self.obstacles:
                    distance = d - 1
                    break
            
            sensors.append(distance / self.sensor_range)
        
        return sensors
    
    def get_state_size(self) -> int:
        """Get the size of the state vector."""
        base_size = 4  # agent_x, agent_y, goal_x, goal_y
        if self.use_sensors:
            base_size += 4  # 4 directional sensors
        return base_size
    
    def get_action_space(self) -> List[int]:
        """Get list of valid actions."""
        return list(self.ACTIONS.keys())
    
    def get_action_size(self) -> int:
        """Get number of possible actions."""
        return len(self.ACTIONS)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this template."""
        return {
            "grid_size": [5, 5],
            "obstacle_count": 3,
            "max_steps": 100,
            "use_sensors": True,
            "sensor_range": 2,
        }
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get JSON schema for template configuration."""
        return {
            "type": "object",
            "properties": {
                "grid_size": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 3},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Grid dimensions [rows, cols]",
                },
                "obstacle_count": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of obstacles in the grid",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 10,
                    "description": "Maximum steps per episode",
                },
                "use_sensors": {
                    "type": "boolean",
                    "description": "Whether to include obstacle sensors in state",
                },
                "sensor_range": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Range of obstacle sensors",
                },
            },
            "required": [],
        }
    
    def render(self) -> str:
        """Render environment state as ASCII art."""
        grid = []
        for r in range(self.grid_size[0]):
            row = []
            for c in range(self.grid_size[1]):
                pos = (r, c)
                if pos == self.agent_pos:
                    row.append("A")
                elif pos == self.goal_pos:
                    row.append("G")
                elif pos in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            grid.append(" ".join(row))
        return "\n".join(grid)
