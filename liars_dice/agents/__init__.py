"""
Central agent registry and registration decorator for Liar's Dice agents.
Use @register_agent("name") above your agent class to make it available for experiments and CLI.
All agent modules must be imported here to ensure registration occurs.
"""

AGENT_MAP = {}

def register_agent(name):
	"""
	Decorator to register an agent class under a given name.
	Usage:
		@register_agent("random")
		class RandomAgent(Agent): ...
	"""
	def decorator(cls):
		AGENT_MAP[name] = cls
		return cls
	return decorator

# Import all agent modules here to ensure registration decorators run
from . import random_agent

