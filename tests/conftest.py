import numpy as np
import pytest


def pytest_configure(config):
    # This sets the precision globally for the entire test session
    np.set_printoptions(precision=3, suppress=True)
    # np.set_printoptions(formatter={"float_kind": lambda x: f"{x:0.3f}"})
