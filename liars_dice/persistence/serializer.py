import json
from typing import Any


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)))


def loads(s: str):
    return json.loads(s)

