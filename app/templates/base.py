"""
Base template class and registry for pluggable templates.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Tuple
import numpy as np


class BaseTemplate(ABC):
    """
    Abstract base class for all RL environment templates.
    
    Each template defines:
    - Environment dynamics
    - State space
    - Action space
    - Reward function
    - Configuration schema
    """
    
    # Template metadata - must be overridden by subclasses
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize template with configuration.
        
        Args:
            config: Template-specific configuration dictionary
        """
        self.config = config
        self._validate_config()
        self._setup_environment()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate template configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def _setup_environment(self) -> None:
        """Setup the environment based on configuration."""
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state as numpy array
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass
    
    @abstractmethod
    def get_state_size(self) -> int:
        """Get the size of the state vector."""
        pass
    
    @abstractmethod
    def get_action_space(self) -> List[int]:
        """Get list of valid actions."""
        pass
    
    @abstractmethod
    def get_action_size(self) -> int:
        """Get number of possible actions."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this template."""
        pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for template configuration.
        
        Returns:
            Dictionary describing configuration schema
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }
    
    def render(self) -> Optional[str]:
        """
        Optional: Render environment state as string for debugging.
        
        Returns:
            String representation of current state
        """
        return None
    
    def close(self) -> None:
        """Cleanup resources if needed."""
        pass


class TemplateRegistry:
    """
    Registry for discovering and accessing templates.
    
    Uses a class-based registry pattern for extensibility.
    """
    
    _templates: Dict[str, Type[BaseTemplate]] = {}
    
    @classmethod
    def register(cls, template_class: Type[BaseTemplate]) -> None:
        """
        Register a template class.
        
        Args:
            template_class: Template class to register
            
        Raises:
            ValueError: If template name is empty or already registered
        """
        if not template_class.name:
            raise ValueError(f"Template class {template_class.__name__} must have a name")
        
        if template_class.name in cls._templates:
            raise ValueError(f"Template '{template_class.name}' is already registered")
        
        cls._templates[template_class.name] = template_class
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a template.
        
        Args:
            name: Name of template to unregister
        """
        if name in cls._templates:
            del cls._templates[name]
    
    @classmethod
    def get(cls, name: str) -> Type[BaseTemplate]:
        """
        Get a template class by name.
        
        Args:
            name: Template name
            
        Returns:
            Template class
            
        Raises:
            ValueError: If template not found
        """
        if name not in cls._templates:
            available = ", ".join(cls.list_templates())
            raise ValueError(f"Template '{name}' not found. Available: {available}")
        
        return cls._templates[name]
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseTemplate:
        """
        Create a template instance.
        
        Args:
            name: Template name
            config: Template configuration
            
        Returns:
            Template instance
        """
        template_class = cls.get(name)
        return template_class(config)
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """
        List all registered template names.
        
        Returns:
            List of template names
        """
        return list(cls._templates.keys())
    
    @classmethod
    def get_template_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a template.
        
        Args:
            name: Template name
            
        Returns:
            Template metadata
        """
        template_class = cls.get(name)
        return {
            "name": template_class.name,
            "description": template_class.description,
            "version": template_class.version,
            "default_config": template_class({}).get_default_config(),
            "config_schema": template_class({}).get_config_schema(),
        }
    
    @classmethod
    def get_all_templates_info(cls) -> List[Dict[str, Any]]:
        """
        Get information about all registered templates.
        
        Returns:
            List of template metadata
        """
        return [cls.get_template_info(name) for name in cls.list_templates()]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered templates (mainly for testing)."""
        cls._templates.clear()
