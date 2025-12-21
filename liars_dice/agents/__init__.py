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

# Automatically import all agent modules in this directory to ensure registration decorators run
import importlib
import os
import pkgutil

_this_dir = os.path.dirname(__file__)
_pkg_name = __name__
for _, modname, ispkg in pkgutil.iter_modules([_this_dir]):
	if not ispkg and modname not in ("__init__", "base"):
		importlib.import_module(f"{_pkg_name}.{modname}")

