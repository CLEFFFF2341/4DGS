from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Method:
    name: str
    selector: Callable


_METHODS: dict[str, Method] = {}


def register(name: str):
    def decorator(selector: Callable) -> Callable:
        if name in _METHODS:
            raise KeyError(f"Method already registered: {name}")
        _METHODS[name] = Method(name=name, selector=selector)
        return selector
    return decorator


def get(name: str) -> Method:
    return _METHODS[name]


def names() -> list[str]:
    return sorted(_METHODS)


@register("keep_all")
def keep_all(adapter, **_):
    return {
        "static_indices": list(range(adapter.static_count)),
        "dynamic_indices": list(range(adapter.dynamic_count)),
    }
