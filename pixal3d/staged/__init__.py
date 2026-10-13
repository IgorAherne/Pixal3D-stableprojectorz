"""Pixal3D in little memory: the cascade run part by part over a batch of props, each network on the card only
while its part runs (docs/staged_engine.md).

    from pixal3d.staged import StagedEngine, EngineOptions, PropRequest

Names are imported when first asked for, as in pixal3d.pipelines and pixal3d.models: the requests, events and
exceptions (types.py) need no torch.
"""

import importlib

__attributes = {
    'StagedEngine': 'engine',
    'EngineOptions': 'types',
    'Hooks': 'types',
    'PropRequest': 'types',
    'Prepared': 'types',
    'DecodedMesh': 'types',
    'Event': 'types',
    'FinishSteps': 'types',
    'PropFailed': 'types',
    'Cancelled': 'types',
    'BatchBroken': 'types',
    'PARTS': 'types',
    'NetworkLoader': 'loader',
    'PropStore': 'store',
}

__all__ = list(__attributes.keys())


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module = importlib.import_module(f".{__attributes[name]}", __name__)
            globals()[name] = getattr(module, name)
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .engine import StagedEngine
    from .types import (EngineOptions, Hooks, PropRequest, Prepared, DecodedMesh, Event, PropFailed, Cancelled,
                        BatchBroken, PARTS)
    from .loader import NetworkLoader
    from .store import PropStore
