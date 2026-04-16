"""
Template system for Genesys platform.

Templates define the environment, state space, action space, and reward function
for different types of reinforcement learning problems.
"""

from app.templates.base import BaseTemplate, TemplateRegistry
from app.templates.grid_world import GridWorldTemplate
from app.templates.decision_optimization import DecisionOptimizationTemplate

# Register default templates
TemplateRegistry.register(GridWorldTemplate)
TemplateRegistry.register(DecisionOptimizationTemplate)

__all__ = [
    "BaseTemplate",
    "TemplateRegistry",
    "GridWorldTemplate",
    "DecisionOptimizationTemplate",
]
