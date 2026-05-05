"""Input validation module

This module contains the input validation functions used in the library.

Notes
-----
For developers: when writing a new decorator, it is important to use the
`functools.wraps` decorator to preserve the decorated function's metadata
(name, docstring, etc.). Otherwise, the decorated function will be relaced
by the decorator's wrapper function, and the metadata will be lost. This
will cause problems with beartype, which relies on the metadata to perform
type checking.
"""

from .converters import convert_inputs

# from .typechecking import typecheck
