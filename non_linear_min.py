import numpy as np
from typing import Callable, Tuple
from rosenbrock import *


def non_linear_min(
    f_orig: Callable[[np.ndarray], float],
    x0: np.ndarray,
    method: str,
    tol: float,
    restart: bool,
    printout: bool,
) -> Tuple[np.ndarray, int, int, float]:
    """
    Nonlinear optimization using DFP or BFGS quasi-Newton methods.

    Args:
        f : Callable
            Objective function
        x0 : np.ndarray
            Initial guess
        method : str
            "DFP" or "BFGS"
        tol : float
            Convergence tolerance for gradient norm
        restart : bool
            If True, reset Hessian approximation to identity if something goes wrong
        printout : bool
            If True, print iteration information

    Returns:
        x : np.ndarray
            Minimum x*
        n_iter : int
            Number of iterations
        n_fval : int
            Number of function evaluations
        gnorm : float
            Norm of function gradient at x*
    """
    calls = [0]

    def f(x):
        calls[0] += 1
        return f_orig(x)

    # Parameters for Wolfe conditions
    c1 = 1e-4
    c2 = 0.9

    # HELPER FUNCTIONS
    norm = np.linalg.norm

    def grad_c(
        f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1.0e-8
    ) -> np.ndarray:
        """
        Calculates the gradient (numpy-1D-array) g of
        function f with central differences
        x is a numpy-1D-array, f(x) is a scalar
        """
        try:
            assert len(x.shape) == 1
        except:
            raise ("grad.py, grad_c: x must be a 1D-numpy-array.")

        inv_2h = 0.5 / h
        lx = x.shape[0]
        g = np.zeros(x.shape, x.dtype)

        for i in range(lx):
            hi = np.zeros(x.shape, x.dtype)
            hi[i] = h
            g[i] = (f(x + hi) - f(x - hi)) * inv_2h

        return g

    def _hesse(s, y, H, mthd) -> np.ndarray:
        if mthd == "dfp":
            Hy = H @ y
            sTy = np.dot(s, y)
            yTHy = np.dot(y, Hy)

            if sTy > 1e-12:
                return H - np.outer(Hy, Hy) / yTHy + np.outer(s, s) / sTy
            return None  # if not pos. def.
        if mthd == "bfgs":
            rho = 1.0 / np.dot(y, s)

            if rho > 0:
                I_minus_rhosyT = np.eye(n) - rho * np.outer(s, y)
                return I_minus_rhosyT @ H @ I_minus_rhosyT.T + rho * np.outer(s, s)
            return None  # if not pos. def.
        raise ValueError(f"Unknown update formula: {mthd}")

    def _line_search(f, x, p, g) -> float:
        """Line search function that satisfies Wolfe conditions, from Nocedal&Wright 2006 p. 59-60
        Args:
            f: Objective function
            x: Current point
            p: Search direction
            fx: Objective function value at x
            g: Objective function gradient at x
        """
        alpha = 1
        phi = lambda alpha: f(x + alpha * p)

        def _zoom(alpha_lo, alpha_hi):
            dphi_0 = np.dot(grad_c(f, x), p)

            for _ in range(max_iter):
                alpha_j = (alpha_lo + alpha_hi) / 2
                phi_j = phi(alpha_j)
                phi_0 = phi(0)
                if phi_j > phi_0 + c1 * alpha_j * dphi_0 or phi_j >= phi(alpha_lo):
                    alpha_hi = alpha_j
                else:
                    dphi_j = np.dot(grad_c(f, x + alpha_j * p), p)
                    dphi_lo = np.dot(grad_c(f, x + alpha_lo * p), p)
                    if np.abs(dphi_j) <= -c2 * dphi_lo:
                        return alpha_j

                    if dphi_j * (alpha_hi - alpha_lo) >= 0:
                        alpha_hi = alpha_lo

                    alpha_lo = alpha_j

            return alpha_lo

        dphi_0 = np.dot(g, p)
        alpha_prev = 0.0
        phi_prev = phi_0 = phi(0)

        for i in range(max_iter):
            phi_i = phi(alpha)

            if phi_i > phi_0 + c1 * alpha * dphi_0 or (i > 0 and phi_i >= phi_prev):
                return _zoom(alpha_prev, alpha)

            dphi_i = np.dot(grad_c(f, x + alpha * p), p)

            if np.abs(dphi_i) <= -c2 * dphi_0:
                return alpha

            if dphi_i >= 0:
                return _zoom(alpha, alpha_prev)

            alpha_prev = alpha
            phi_prev = phi_i
            alpha = alpha * 1.5

        return alpha

    x = x0.copy()
    n = len(x)

    g = grad_c(f, x)

    H = np.eye(n)

    n_iter = 0
    max_iter = 1000

    if printout:
        print(f"Iteration {n_iter}: f(x) = {f(x):.6e}, ||g|| = {np.linalg.norm(g):.6e}")

    # Main optimization loop
    while norm(g) > tol and n_iter < max_iter:
        p = -H @ g
        alpha = _line_search(f, x, p, g)
        x_new = x + alpha * p
        g_new = grad_c(f, x_new)

        s = alpha * p
        y = g_new - g

        H = _hesse(s, y, H, method)
        if H is None:
            if restart:
                H = np.eye(n)
            else:
                raise RuntimeError("Hesse matrix is not pos. def, consider restarting")

        x = x_new
        g = g_new
        n_iter += 1

        if printout:
            print(
                f"Iteration {n_iter}: f(x) = {f(x):.6e}, ||g|| = {np.linalg.norm(g):.6e}, alpha = {alpha}"
            )
    return (
        x,
        calls[0],
        n_iter,
        f(x),
    )


if __name__ == "__main__":
    f = lambda x: np.sum(x**2)
    print(non_linear_min(rosenbrock, np.random.rand(2) * 100, "bfgs", 1e-5, True, True))
