from pathlib import Path
import importlib
from typing import Any, Union, Dict, Sequence
import torch
from torch import nn


def dump_object(
    obj: Any, args: Sequence[Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    module_name = obj.__class__.__module__
    class_name = obj.__class__.__qualname__

    state = {
        "__kanachan__": "11fc2bfe-c4c7-402e-b11e-7cb3ff6f9945",
        "module": module_name,
        "class": class_name,
    }
    if len(args) != 0:
        state["args"] = args
    if len(kwargs) != 0:
        state["kwargs"] = kwargs

    return state


def dump_model(
    model: nn.Module, args: Sequence[Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    state = dump_object(model, args, kwargs)
    state["state_dict"] = model.state_dict()

    return state


def _load_model(state: Any, map_location: torch.device | None) -> Any:
    if not isinstance(state, dict):
        return state
    if "__kanachan__" not in state:
        return state
    if state["__kanachan__"] != "11fc2bfe-c4c7-402e-b11e-7cb3ff6f9945":
        return state

    if "module" not in state:
        errmsg = "A broken Kanachan's model file."
        raise RuntimeError(errmsg)
    module_name = state["module"]
    module = importlib.import_module(module_name)

    if "class" not in state:
        errmsg = "A broken Kanachan's model file."
        raise RuntimeError(errmsg)
    class_name = state["class"]
    if not hasattr(module, class_name):
        errmsg = (
            f"The `{module_name}` mudule does not have the `{class_name}`"
            " class."
        )
        raise RuntimeError(errmsg)
    _class = getattr(module, class_name)

    args = []
    if "args" in state:
        for arg in state["args"]:
            args.append(_load_model(arg, map_location))

    kwargs = {}
    if "kwargs" in state:
        for name, value in state["kwargs"].items():
            kwargs[name] = _load_model(value, map_location)
    if map_location is not None and "device" in kwargs:
        kwargs["device"] = map_location

    model: nn.Module = _class(*args, **kwargs)

    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])

    return model


def load_model(
    model_path: Union[str, Path], map_location: torch.device | None = None
) -> nn.Module:
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if not model_path.exists():
        errmsg = f"{model_path}: Does not exist."
        raise RuntimeError(errmsg)
    if not model_path.is_file():
        errmsg = f"{model_path}: Not a file."
        raise RuntimeError(errmsg)

    state = torch.load(model_path, map_location=map_location)
    if not isinstance(state, dict):
        errmsg = f"{model_path}: Not a Kanachan's model file."
        raise RuntimeError(errmsg)
    if "__kanachan__" not in state:
        errmsg = f"{model_path}: Not a Kanachan's model file."
        raise RuntimeError(errmsg)
    if state["__kanachan__"] != "11fc2bfe-c4c7-402e-b11e-7cb3ff6f9945":
        errmsg = f"{model_path}: Not a Kanachan's model file."
        raise RuntimeError(errmsg)

    return _load_model(state, map_location)
