from typing import Callable
import numpy as np


def grad_c(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-8,
) -> np.ndarray:
    """
    central-difference gradient for scalar f.

    uses a relative step per component: hi = h * (1 + |x_i|)
    this reduces cancellation when |x| is large and improves convergence for small tol.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:  #robust input check
        raise TypeError("grad.py, grad_c: x must be a 1D numpy array.")
    if h <= 0:
        raise ValueError("grad.py, grad_c: h must be positive.")

    g = np.zeros_like(x, dtype=float)

    for i in range(x.size):
        hi = h * (1.0 + abs(x[i]))  #relative step (key fix)
        e = np.zeros_like(x)
        e[i] = 1.0
        g[i] = (f(x + hi * e) - f(x - hi * e)) / (2.0 * hi)

    return g


def jacobian_c(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-8,
) -> np.ndarray:
    """
    central-difference jacobian for vector-valued f.
    same relative-step logic per component.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise TypeError("grad.py, jacobian_c: x must be a 1D numpy array.")
    if h <= 0:
        raise ValueError("grad.py, jacobian_c: h must be positive.")

    fx = np.asarray(f(x), dtype=float)
    if fx.ndim != 1:
        raise TypeError("grad.py, jacobian_c: f(x) must be a 1D numpy array.")

    J = np.zeros((fx.size, x.size), dtype=float)

    for i in range(x.size):
        hi = h * (1.0 + abs(x[i]))  #relative step
        e = np.zeros_like(x)
        e[i] = 1.0
        J[:, i] = (np.asarray(f(x + hi * e), dtype=float) - np.asarray(f(x - hi * e), dtype=float)) / (2.0 * hi)

    return J
