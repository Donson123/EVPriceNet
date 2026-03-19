from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Type

@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapter_cls: Type

_REG: Dict[str, ModelSpec] = {}

def register(name: str, adapter_cls: Type) -> None:
    if name in _REG:
        raise ValueError(f"Model already registered: {name}")
    _REG[name] = ModelSpec(name=name, adapter_cls=adapter_cls)

def all_models() -> Dict[str, ModelSpec]:
    return dict(_REG)

def clear() -> None:
    """
    Clear the global model registry.
    we want to re-register models per experimental setup.
    """
    _REG.clear()