
from typing import Callable
import numpy as np



def grad_c(f : Callable[[np.ndarray], float], \
            x : np.ndarray, \
            h : float = 1.e-8) -> np.ndarray:
    """
    Calculates the gradient (numpy-1D-array) g of 
    function f with central differences
    x is a numpy-1D-array, f(x) is a scalar
    """
    try:
        assert(len(x.shape) == 1)
    except:
        raise("grad.py, grad_c: x must be a 1D-numpy-array.")
    
    inv_2h = 0.5 / h
    lx = x.shape[0]
    g = np.zeros(x.shape, x.dtype)

    for i in range(lx):
        hi = np.zeros(x.shape, x.dtype)
        hi[i] = h
        g[i] = (f(x+hi) - f(x-hi)) * inv_2h

    return g



def jacobian_c(f : Callable[[np.ndarray], np.ndarray], \
                x : np.ndarray, \
                h : float = 1.e-8) -> np.ndarray:
    """
    Calculates the jacobian (numpy-2D-array) J of 
    function f with central differences
    x is a numpy-1D-array, f(x) is a numpy-1D-array
    """

    try:
        assert(len(x.shape) == 1)
    except:
        raise("jacobian.py, jacobian_c: x must be a 1D-numpy-array.")
    
    inv_2h = 0.5 / h
    lx = x.shape[0]
    lf = f(x).shape[0]
    J = np.zeros((lf, lx), x.dtype)

    for i in range(lx):
        hi = np.zeros(x.shape, x.dtype)
        hi[i] = h
        J[:,i] = (f(x+hi) - f(x-hi)) * inv_2h

    return J
