from pathlib import Path
import importlib
from typing import Any, Union, Dict, Iterable
import torch
from torch import nn


def dump_object(obj: Any, args: Iterable[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    module_name = obj.__class__.__module__
    class_name = obj.__class__.__qualname__

    state = {
        '__kanachan__': '11fc2bfe-c4c7-402e-b11e-7cb3ff6f9945',
        'module': module_name,
        'class': class_name,
    }
    if len(args) != 0:
        state['args'] = args
    if len(kwargs) != 0:
        state['kwargs'] = kwargs

    return state


def dump_model(model: nn.Module, args: Iterable[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    state = dump_object(model, args, kwargs)
    state['state_dict'] = model.state_dict()

    return state


def _load_model(state: Any) -> Any:
    if not isinstance(state, dict):
        return state
    if '__kanachan__' not in state:
        return state
    if state['__kanachan__'] != '11fc2bfe-c4c7-402e-b11e-7cb3ff6f9945':
        return state

    if 'module' not in state:
        raise RuntimeError('A broken Kanachan\'s model file.')
    module_name = state['module']
    module = importlib.import_module(module_name)

    if 'class' not in state:
        raise RuntimeError('A broken Kanachan\'s model file.')
    class_name = state['class']
    if not hasattr(module, class_name):
        raise RuntimeError(f'The `{module_name}` mudule does not have the `{class_name}` class.')
    _class = getattr(module, class_name)

    args = []
    if 'args' in state:
        for arg in state['args']:
            args.append(_load_model(arg))

    kwargs = {}
    if 'kwargs' in state:
        for name, value in state['kwargs'].items():
            kwargs[name] = _load_model(value)

    model: nn.Module = _class(*args, **kwargs)

    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'])

    return model


def load_model(model_path: Union[str, Path], map_location=None) -> nn.Module:
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(f'{model_path}: Does not exist.')
    if not model_path.is_file():
        raise RuntimeError(f'{model_path}: Not a file.')

    state = torch.load(model_path, map_location=map_location)
    if not isinstance(state, dict):
        raise RuntimeError(f'{model_path}: Not a Kanachan\'s model file.')
    if '__kanachan__' not in state:
        raise RuntimeError(f'{model_path}: Not a Kanachan\'s model file.')
    if state['__kanachan__'] != '11fc2bfe-c4c7-402e-b11e-7cb3ff6f9945':
        raise RuntimeError(f'{model_path}: Not a Kanachan\'s model file.')

    return _load_model(state)
