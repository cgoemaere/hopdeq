import inspect
from functools import partial
from types import ModuleType
from typing import Callable, Optional


def get_function_from_package(
    package: ModuleType, name: str, kwargs: Optional[dict] = None
) -> Callable:
    """
    Helper function to get a function from a package by using its name
    in string format (and fill its arguments with kwargs)
    """
    function_dict = dict(inspect.getmembers(package, inspect.isfunction))

    try:
        function = function_dict[name]
    except KeyError:
        raise KeyError(
            f"""Received {name} as function, which is not present in package {package.__name__}.\n
                Options are: {list(function_dict.keys())}"""
        )

    if kwargs is not None:
        return partial(function, **kwargs)
    else:
        return function
