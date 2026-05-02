import functools


def cache_clear(self):
    """Reload all cached properties."""
    cls = self.__class__
    # Below, the third argument to getattr is there to return a default value
    # instead of throwing an AttributeError.
    attrs = [
        a
        for a in dir(self)
        if isinstance(getattr(cls, a, cls), functools.cached_property)
    ]
    for a in attrs:
        self.__dict__.pop(a, None)
        # delattr(self, a)

    if hasattr(self, "_cached_methods"):
        for a in self._cached_methods:
            cached_method = getattr(self, a)
            # N.B.: in the __init__ of PolyData, cached methods may not have been
            #       set to an actual cached method yet, with a cache_clear method.
            if hasattr(cached_method, "cache_clear"):
                cached_method.cache_clear()

    if hasattr(self, "_cached_properties"):
        for a in self._cached_properties:
            if hasattr(self, "_cached_" + a):
                delattr(self, "_cached_" + a)


def immutable_cached_property(*, function, cache):
    """This decorator is roughly equivalent to @cached_property, but better suited here.

    Notably, it does not allow the cached property to be set to a new value,
    and allows the docstrings to be discovered by pytest.
    """

    def cached_func(self):
        if not cache:
            return function(self)
        else:
            if not hasattr(self, "_cached" + function.__name__):
                setattr(self, "_cached" + function.__name__, function(self))
            return getattr(self, "_cached" + function.__name__)

    return property(cached_func)


def add_cached_methods_to_sphinx(cls):
    """Ensures that e.g. ``PolyData.point_normals`` is documented in Sphinx.

    Cached methods are instance methods that are memoized with ``functools.lru_cache``.
    This small decorator ensures that although ``PolyData.point_normals`` is a cached
    front-end for the private method ``PolyData._point_normals`` that is instantiated
    in the `__init__` method, the Sphinx documentation will look as though
    it was a regular method.
    """
    for method_name in cls._cached_methods + cls._cached_properties:
        # As far as Sphinx is concerned,
        # self.method_name = self._method_name
        # Then, at the end of the __init__, we overwrite self.method_name
        # with a memoized version of self._method_name.
        setattr(cls, method_name, getattr(cls, "_" + method_name))
    return cls


def cache_methods_and_properties(*, cls, instance, cache_size):
    for method_name in instance._cached_methods:
        setattr(
            instance,
            method_name,
            functools.lru_cache(maxsize=cache_size)(
                getattr(instance, "_" + method_name)
            ),
        )

    # Cached properties are not cached if cache_size is 0
    for method_name in instance._cached_properties:
        setattr(
            cls,
            method_name,
            immutable_cached_property(
                function=getattr(cls, "_" + method_name),
                cache=cache_size != 0,
            ),
        )
