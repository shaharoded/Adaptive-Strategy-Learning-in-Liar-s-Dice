
"""
serializer.py
Provides utility functions for serializing and deserializing game events and state to/from JSON.
Used by recorder.py and experiment scripts to save/load data for analysis or replay.
"""

import json
from typing import Any


def dumps(obj: Any) -> str:
    """
    Serialize a Python object (including dataclasses) to a JSON string.
    Args:
        obj: Object to serialize.
    Returns:
        str: JSON string.
    """
    return json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)))


def loads(s: str):
    """
    Deserialize a JSON string to a Python object (dict/list).
    Args:
        s (str): JSON string.
    Returns:
        object: Deserialized Python object.
    """
    return json.loads(s)

