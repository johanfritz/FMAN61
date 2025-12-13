
import numpy as np


def call_counter(func):
    def helper(x):
        helper.calls += 1
        return func(x)
    helper.calls = 0
    return helper
@call_counter
def rosenbrock(x : np.ndarray) -> float:
    """
    Evaluates Rosenbrock's function for a numpy-array x
    with 1 dimension and size 2
    x = [x[0], x[1]]
    """

    val = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    return val
