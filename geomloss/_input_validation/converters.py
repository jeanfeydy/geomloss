"""Converters for arguments."""

import itertools
from functools import wraps
from inspect import isclass, signature
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

import numpy as np


def convert_inputs(*param_names):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Iterate through the function's parameters and type hints
            for param_name in param_names:

                # Do not waste time on default arguments
                if param_name in bound_args.arguments:

                    # At this point, we know that the parameter has been set
                    # and that it is supposed to be a NumPy array or torch Tensor.
                    # If it is a list or a tuple, we convert it to a NumPy array
                    # of float64.
                    value = bound_args.arguments[param_name]

                    # We attempt to convert lists, tuples, numpy arrays and torch tensors
                    # that do not have the correct dtype.
                    if isinstance(value, list | tuple):
                        value = np.array(value, dtype=np.float64)
                        bound_args.arguments[param_name] = value

                    # Note that other types of "value" (e.g. strings) are not converted

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator
