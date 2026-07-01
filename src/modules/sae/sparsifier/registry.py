from typing import Dict, Type, Any

from .base import BaseSparsifier
from .relu import ReLUSparsifier
from .topk import TopKSparsifier
from .jump_relu import JumpReLUSparsifier

_SPARSIFIER_REGISTRY: Dict[str, Type[BaseSparsifier]] = {
    "relu": ReLUSparsifier,
    "topk": TopKSparsifier,
    "jump_relu": JumpReLUSparsifier,
}


def register_sparsifier(name: str):
    def deco(cls: Type[BaseSparsifier]) -> Type[BaseSparsifier]:
        _SPARSIFIER_REGISTRY[name] = cls
        return cls
    return deco

def get_sparsifier(name: str, **kwargs: Any) -> BaseSparsifier:
    if name not in _SPARSIFIER_REGISTRY:
        raise KeyError(f"Unknown sparsifier: {name}. Available: {list(_SPARSIFIER_REGISTRY)}")
    return _SPARSIFIER_REGISTRY[name](**kwargs)


