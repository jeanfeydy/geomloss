def cast_input(**kwargs):
    """Checks that all arguments have the correct shape, dtype and belong to the same device.

    Returns:
        [type]: [description]
    """
    if kwargs == {}:
        return {}

    for key, (array, shape) in kwargs.items():
        pass
