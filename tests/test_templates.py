"""
Tests for environment templates.
"""

import pytest
import numpy as np

from app.templates.grid_world import GridWorldTemplate
from app.templates.decision_optimization import DecisionOptimizationTemplate
from app.templates.base import TemplateRegistry


class TestGridWorldTemplate:
    """Tests for GridWorld template."""
    
    def test_default_initialization(self):
        """Test template initialization with default config."""
        template = GridWorldTemplate({})
        
        assert template.grid_size == (5, 5)
        assert template.get_action_size() == 4
        assert template.get_state_size() == 8  # 4 base + 4 sensors
    
    def test_custom_config(self):
        """Test template with custom configuration."""
        config = {
            "grid_size": [8, 8],
            "obstacle_count": 5,
            "use_sensors": False,
        }
        template = GridWorldTemplate(config)
        
        assert template.grid_size == (8, 8)
        assert template.obstacle_count == 5
        assert template.get_state_size() == 4  # No sensors
    
    def test_reset(self):
        """Test environment reset."""
        template = GridWorldTemplate({})
        state = template.reset()
        
        assert isinstance(state, np.ndarray)
        assert len(state) == template.get_state_size()
        assert template.agent_pos == (0, 0)
    
    def test_step(self):
        """Test environment step."""
        template = GridWorldTemplate({})
        template.reset()
        
        state, reward, done, info = template.step(1)  # Move RIGHT
        
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "steps" in info
    
    def test_invalid_action(self):
        """Test that invalid actions raise error."""
        template = GridWorldTemplate({})
        template.reset()
        
        with pytest.raises(ValueError):
            template.step(99)


class TestDecisionOptimizationTemplate:
    """Tests for DecisionOptimization template."""
    
    def test_default_initialization(self):
        """Test template initialization with default config."""
        template = DecisionOptimizationTemplate({})
        
        assert template.state_size == 10
        assert template.get_action_space() == [0, 1, 2]
        assert template.get_action_size() == 3
    
    def test_custom_config(self):
        """Test template with custom configuration."""
        config = {
            "state_size": 20,
            "action_space": [0, 1, 2, 3, 4],
            "reward_type": "quadratic",
        }
        template = DecisionOptimizationTemplate(config)
        
        assert template.state_size == 20
        assert template.get_action_space() == [0, 1, 2, 3, 4]
        assert template.reward_type == "quadratic"
    
    def test_reset(self):
        """Test environment reset."""
        template = DecisionOptimizationTemplate({})
        state = template.reset()
        
        assert isinstance(state, np.ndarray)
        assert len(state) == template.state_size
    
    def test_step(self):
        """Test environment step."""
        template = DecisionOptimizationTemplate({})
        template.reset()
        
        state, reward, done, info = template.step(1)
        
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "is_optimal_action" in info


class TestTemplateRegistry:
    """Tests for template registry."""
    
    def test_list_templates(self):
        """Test listing registered templates."""
        templates = TemplateRegistry.list_templates()
        
        assert "grid_world" in templates
        assert "decision_optimization" in templates
    
    def test_get_template(self):
        """Test getting template class."""
        template_class = TemplateRegistry.get("grid_world")
        
        assert template_class == GridWorldTemplate
    
    def test_get_invalid_template(self):
        """Test getting non-existent template raises error."""
        with pytest.raises(ValueError):
            TemplateRegistry.get("non_existent")
    
    def test_create_template(self):
        """Test creating template instance."""
        template = TemplateRegistry.create("grid_world", {})
        
        assert isinstance(template, GridWorldTemplate)
    
    def test_get_template_info(self):
        """Test getting template information."""
        info = TemplateRegistry.get_template_info("grid_world")
        
        assert "name" in info
        assert "description" in info
        assert "default_config" in info
